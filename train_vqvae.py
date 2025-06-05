import os
import sys
os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'
import torch
import torch.multiprocessing as mp
import random
import librosa
import yaml
import argparse
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import glob
from tqdm import tqdm
import shutil
from pathlib import Path
from random import randint

from modules.commons import recursive_munch, build_model, load_checkpoint2, MyModel
from optimizers import build_optimizer
from data.ft_dataset import build_ft_dataloader,FT_Dataset,collate
from hf_utils import load_custom_model_from_hf
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist 

from omegaconf import OmegaConf
from modules.vqvae.xtts_dvae import DiscreteVAE
import re

def load_checkpoint(model: torch.nn.Module, model_pth: str) -> dict:
    checkpoint = torch.load(model_pth, map_location='cpu')
    checkpoint = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(checkpoint, strict=True)
    info_path = re.sub('.pth$', '.yaml', model_pth)
    configs = {}
    if os.path.exists(info_path):
        with open(info_path, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
    return configs


class Trainer:
    def __init__(self,
                 config_path,
                 pretrained_ckpt_path,
                 data_dir,
                 exp_dir,
                 batch_size=0,
                 num_workers=0,
                 steps=1000,
                 save_interval=500,
                 max_epochs=1000,
                 device = 'cuda:0',
                 local_rank=0,
                 n_gpu=1
                 ):
        self.local_rank = local_rank
        self.n_gpu = n_gpu
        self.device = device
        
        config = yaml.safe_load(open(config_path))
        self.log_dir = Path(exp_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        # copy config file to log dir
        shutil.copyfile(config_path, os.path.join(self.log_dir, os.path.basename(config_path)))
        batch_size = config.get('batch_size', 10) if batch_size == 0 else batch_size
        self.max_steps = steps

        self.n_epochs = max_epochs
        self.log_interval = config.get('log_interval', 10)
        self.save_interval = save_interval

        self.sr = config['dataset']['mel'].get('sample_rate', 22050)
        
        # DataLoader with DistributedSampler
        self.train_dataset = FT_Dataset(config['dataset']['training_files'], config['dataset']['mel'])
        
        self.train_sampler = DistributedSampler(self.train_dataset) if self.n_gpu > 1 else None
        
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate,
            sampler=self.train_sampler,
            drop_last=True
        )
        self.f0_condition = config['model_params']['DiT'].get('f0_condition', False)
        self.build_sv_model(device, config)
        self.build_semantic_fn(device, config)
        if self.f0_condition:
            self.build_f0_fn(device, config)
#         self.build_converter(device, config)
        self.build_vocoder(device, config)

        scheduler_params = {
            "warmup_steps": 0,
            "base_lr": 0.00001,
        }

        self.model_params = recursive_munch(config['model_params'])
        self.model = MyModel(self.model_params)
        self.model = self.model.to(device)
        
        self.model.models['cfm'].estimator.setup_caches(max_batch_size=batch_size, max_seq_length=8192)
        
        self.optimizer = build_optimizer(self.model.models,
                                         lr=float(scheduler_params['base_lr']))
    
        if pretrained_ckpt_path is None:
            # find latest checkpoint
            available_checkpoints = glob.glob(os.path.join(self.log_dir, "DiT_epoch_*_step_*.pth"))
            if len(available_checkpoints) > 0:
                latest_checkpoint = max(
                    available_checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0])
                )
                earliest_checkpoint = min(
                    available_checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0])
                )
                # delete the earliest checkpoint if we have more than 2
                if (
                    earliest_checkpoint != latest_checkpoint
                    and len(available_checkpoints) > 2
                ):
                    os.remove(earliest_checkpoint)
                    print(f"Removed {earliest_checkpoint}")
            elif config.get('pretrained_model', ''):
                latest_checkpoint = load_custom_model_from_hf("Plachta/Seed-VC", config['pretrained_model'], None)
            else:
                latest_checkpoint = ""
        else:
            assert os.path.exists(pretrained_ckpt_path), f"Pretrained checkpoint {pretrained_ckpt_path} not found"
            latest_checkpoint = pretrained_ckpt_path

        if os.path.exists(latest_checkpoint):
            print(latest_checkpoint)
            self.model, self.optimizer, self.epoch, self.iters = load_checkpoint2(
                self.model, self.optimizer, latest_checkpoint,
                load_only_params=True,
                ignore_modules=[],
                is_distributed=False
            )
            print(f"Loaded checkpoint from {latest_checkpoint}")
        else:
            self.epoch, self.iters = 0, 0
            print("Failed to load any checkpoint, training from scratch.")

        ## load vqvae model ##
        self.dvae = DiscreteVAE(**config['vqvae'])
        dvae_path = config['dvae_checkpoint']
        load_checkpoint(self.dvae, dvae_path)
        self.dvae.eval()
        self.dvae = self.dvae.to(device)
        print('load dvae from {}'.format(dvae_path))
        
        if n_gpu > 1:
            self.model = DDP(self.model, device_ids=[local_rank], find_unused_parameters=True)

    def build_sv_model(self, device, config):
        from modules.campplus.DTDNN import CAMPPlus
        self.campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus_sd_path = load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin", config_filename=None)
        campplus_sd = torch.load(campplus_sd_path, map_location='cpu')
        self.campplus_model.load_state_dict(campplus_sd)
        self.campplus_model.eval()
        self.campplus_model.to(device)
        self.sv_fn = self.campplus_model

    def build_f0_fn(self, device, config):
        from modules.rmvpe import RMVPE
        model_path = load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt", None)
        self.rmvpe = RMVPE(model_path, is_half=False, device=device)
        self.f0_fn = self.rmvpe

    def build_converter(self, device, config):
        from modules.openvoice.api import ToneColorConverter
        ckpt_converter, config_converter = load_custom_model_from_hf("myshell-ai/OpenVoiceV2", "converter/checkpoint.pth", "converter/config.json")
        self.tone_color_converter = ToneColorConverter(config_converter, device=device)
        self.tone_color_converter.load_ckpt(ckpt_converter)
        self.tone_color_converter.model.eval()
        se_db_path = load_custom_model_from_hf("Plachta/Seed-VC", "se_db.pt", None)
        self.se_db = torch.load(se_db_path, map_location='cpu')

    def build_vocoder(self, device, config):
        vocoder_type = config['model_params']['vocoder']['type']
        vocoder_name = config['model_params']['vocoder'].get('name', None)
        if vocoder_type == 'bigvgan':
            from modules.bigvgan import bigvgan
            self.bigvgan_model = bigvgan.BigVGAN.from_pretrained(vocoder_name, use_cuda_kernel=False)
            self.bigvgan_model.remove_weight_norm()
            self.bigvgan_model = self.bigvgan_model.eval().to(device)
            vocoder_fn = self.bigvgan_model
        elif vocoder_type == 'hifigan':
            from modules.hifigan.generator import HiFTGenerator
            from modules.hifigan.f0_predictor import ConvRNNF0Predictor
            hift_config = yaml.safe_load(open('configs/hifigan.yml', 'r'))
            hift_path = load_custom_model_from_hf("FunAudioLLM/CosyVoice-300M", 'hift.pt', None)
            self.hift_gen = HiFTGenerator(**hift_config['hift'],
                                          f0_predictor=ConvRNNF0Predictor(**hift_config['f0_predictor']))
            self.hift_gen.load_state_dict(torch.load(hift_path, map_location='cpu'))
            self.hift_gen.eval()
            self.hift_gen.to(device)
            vocoder_fn = self.hift_gen
        else:
            raise ValueError(f"Unsupported vocoder type: {vocoder_type}")
        self.vocoder_fn = vocoder_fn

    def build_semantic_fn(self, device, config):
        speech_tokenizer_type = config['model_params']['speech_tokenizer'].get('type', 'cosyvoice')
        if speech_tokenizer_type == 'wav2vecbert':
            from transformers import (
                SeamlessM4TFeatureExtractor,
                Wav2Vec2BertModel,
            )
            # Load semantic model
            self.feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(config['dataset']['semantic_model_path'])
            self.semantic_model = Wav2Vec2BertModel.from_pretrained(config['dataset']['semantic_model_path'])
            self.semantic_model.eval().to(device)
            self.stat_mean_var = torch.load(config['dataset']['semantic_mean_var_path'])
            self.semantic_mean = self.stat_mean_var["mean"].to(device)
            self.semantic_std = torch.sqrt(self.stat_mean_var["var"]).to(device)
            def semantic_fn(waves_16k):
                speech = waves_16k.cpu().numpy()
                output = self.feature_extractor(speech, return_attention_mask=True)
                input_features = torch.from_numpy(output["input_features"]).to(device)
                attention_mask = torch.from_numpy(output["attention_mask"]).to(device)
                with torch.no_grad():
                    outputs = self.semantic_model(
                                input_features=input_features,
                                attention_mask=attention_mask,
                                output_hidden_states=True,
                    )
                    emb = outputs.hidden_states[17]  # (B, T, C)
                    emb = (emb - self.semantic_mean) / self.semantic_std
                    emb = emb * attention_mask.unsqueeze(-1)
                    emb = emb.transpose(1, 2)
                    feat, code = self.dvae.get_codebook_indices(emb)
                return feat #[1,620,1024] 
        
        elif speech_tokenizer_type == 'whisper':
            from transformers import AutoFeatureExtractor, WhisperModel
            whisper_model_name = config['model_params']['speech_tokenizer']['name']
            self.whisper_model = WhisperModel.from_pretrained(whisper_model_name).to(device)
            self.whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_model_name)
            # remove decoder to save memory
            del self.whisper_model.decoder

            def semantic_fn(waves_16k):
                ori_inputs = self.whisper_feature_extractor(
                    [w16k.cpu().numpy() for w16k in waves_16k],
                    return_tensors="pt",
                    return_attention_mask=True,
                    sampling_rate=16000,
                )
                ori_input_features = self.whisper_model._mask_input_features(
                    ori_inputs.input_features, attention_mask=ori_inputs.attention_mask
                ).to(device)
                with torch.no_grad():
                    ori_outputs = self.whisper_model.encoder(
                        ori_input_features.to(self.whisper_model.encoder.dtype),
                        head_mask=None,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                S_ori = ori_outputs.last_hidden_state.to(torch.float32)
                S_ori = S_ori[:, :waves_16k.size(-1) // 320 + 1]
                return S_ori

        elif speech_tokenizer_type == 'xlsr':
            from transformers import (
                Wav2Vec2FeatureExtractor,
                Wav2Vec2Model,
            )
            model_name = config['model_params']['speech_tokenizer']['name']
            output_layer = config['model_params']['speech_tokenizer']['output_layer']
            self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            self.wav2vec_model = Wav2Vec2Model.from_pretrained(model_name)
            self.wav2vec_model.encoder.layers = self.wav2vec_model.encoder.layers[:output_layer]
            self.wav2vec_model = self.wav2vec_model.to(device)
            self.wav2vec_model = self.wav2vec_model.eval()
            self.wav2vec_model = self.wav2vec_model.half()

            def semantic_fn(waves_16k):
                ori_waves_16k_input_list = [waves_16k[bib].cpu().numpy() for bib in range(len(waves_16k))]
                ori_inputs = self.wav2vec_feature_extractor(
                    ori_waves_16k_input_list,
                    return_tensors="pt",
                    return_attention_mask=True,
                    padding=True,
                    sampling_rate=16000
                ).to(device)
                with torch.no_grad():
                    ori_outputs = self.wav2vec_model(
                        ori_inputs.input_values.half(),
                    )
                S_ori = ori_outputs.last_hidden_state.float()
                return S_ori
        else:
            raise ValueError(f"Unsupported speech tokenizer type: {speech_tokenizer_type}")
        self.semantic_fn = semantic_fn

    def train_one_step(self, batch):
        waves, mels, wave_lengths, mel_input_length = batch

        B = waves.size(0)
        target_size = mels.size(2)
        target = mels
        target_lengths = mel_input_length

        # get speaker embedding
        if self.sr != 22050:
            waves_22k = torchaudio.functional.resample(waves, self.sr, 22050)
            wave_lengths_22k = (wave_lengths.float() * 22050 / self.sr).long()
        else:
            waves_22k = waves
            wave_lengths_22k = wave_lengths
#         se_batch = self.tone_color_converter.extract_se(waves_22k, wave_lengths_22k)

#         ref_se_idx = torch.randint(0, len(self.se_db), (B,))
#         ref_se = self.se_db[ref_se_idx].to(self.device)

        waves_16k = torchaudio.functional.resample(waves, self.sr, 16000)
        wave_lengths_16k = (wave_lengths.float() * 16000 / self.sr).long()

        S_ori = self.semantic_fn(waves_16k)
        if self.f0_condition:
            F0_ori = self.rmvpe.infer_from_audio_batch(waves_16k)
        else:
            F0_ori = None

        ori_cond, _, ori_codes, ori_commitment_loss, ori_codebook_loss = (
            self.model.forward2(S_ori, target_lengths, F0_ori)
        )
        if ori_commitment_loss is None:
            ori_commitment_loss = 0
            ori_codebook_loss = 0

        # randomly set a length as prompt
        prompt_len_max = target_lengths - 1
        prompt_len = (torch.rand([B], device=ori_cond.device) * prompt_len_max).floor().long()
        prompt_len[torch.rand([B], device=ori_cond.device) < 0.1] = 0

        # for prompt cond token, use ori_cond instead of alt_cond
        cond = ori_cond.clone()
        
        # diffusion target
        common_min_len = min(target_size, cond.size(1))
        target = target[:, :, :common_min_len]
        cond = cond[:, :common_min_len]
        target_lengths = torch.clamp(target_lengths, max=common_min_len)
        x = target

        # style vectors are extracted from the prompt only
        feat_list = []
        for bib in range(B):
            feat = kaldi.fbank(
                waves_16k[bib:bib + 1, :wave_lengths_16k[bib]],
                num_mel_bins=80,
                dither=0,
                sample_frequency=16000
            )
            feat = feat - feat.mean(dim=0, keepdim=True)
            feat_list.append(feat)
        y_list = []
        with torch.no_grad():
            for feat in feat_list:
                y = self.sv_fn(feat.unsqueeze(0))
                y_list.append(y)
        y = torch.cat(y_list, dim=0)

        loss, _ = self.model.forward(x, target_lengths, prompt_len, cond, y) # x:mel target_lenths: mel_input_len,cond: semantic feature, y:style feature

        loss_total = (
            loss +
            (ori_commitment_loss) * 0.05 +
            (ori_codebook_loss) * 0.15
        )

        self.optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.module.models['cfm'].parameters(), 10.0)
        torch.nn.utils.clip_grad_norm_(self.model.module.models['length_regulator'].parameters(), 10.0)
        self.optimizer.step('cfm')
        self.optimizer.step('length_regulator')
        self.optimizer.scheduler(key='cfm')
        self.optimizer.scheduler(key='length_regulator')

        return loss.detach().item()

    def train_one_epoch(self,rank):
#         _ = [self.model[key].train() for key in self.model]
        self.model.train()
        # Sync the sampler to shuffle differently for each epoch
        for i, batch in enumerate(tqdm(self.train_dataloader)):
            batch = [b.to(self.device) for b in batch]
            loss = self.train_one_step(batch)
            self.ema_loss = (
                self.ema_loss * self.loss_smoothing_rate + loss * (1 - self.loss_smoothing_rate)
                if self.iters > 0 else loss
            )
            if self.iters % self.log_interval == 0:
                if rank==0:
                    print(f"epoch {self.epoch}, step {self.iters}, loss: {self.ema_loss}")
            self.iters += 1

            if self.iters >= self.max_steps:
                break

            if self.iters % self.save_interval == 0:
                if rank==0:
                    print('Saving..')
                    state = {
                        'net': {key: self.model.module.models[key].state_dict() for key in self.model.module.models},
                        'optimizer': self.optimizer.state_dict(),
                        'scheduler': self.optimizer.scheduler_state_dict(),
                        'iters': self.iters,
                        'epoch': self.epoch,
                    }
                    save_path = os.path.join(
                        self.log_dir,
                        f'DiT_epoch_{self.epoch:05d}_step_{self.iters:05d}.pth'
                    )
                    torch.save(state, save_path)

                    # find all checkpoints and remove old ones
                    checkpoints = glob.glob(os.path.join(self.log_dir, 'DiT_epoch_*.pth'))
                    if len(checkpoints) > 2:
                        checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                        for cp in checkpoints[:-2]:
                            os.remove(cp)

    def train(self,rank):
        self.ema_loss = 0
        self.loss_smoothing_rate = 0.99
        for epoch in range(self.n_epochs):
            self.epoch = epoch
            if self.n_gpu > 1:
                self.train_sampler.set_epoch(self.epoch)
            self.train_one_epoch(rank)
            # Save after each epoch
            if rank==0:
                print('Epoch completed. Saving..')
                state = {
                    'net': {key: self.model.module.models[key].state_dict() for key in self.model.module.models},
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.optimizer.scheduler_state_dict(),
                    'iters': self.iters,
                    'epoch': self.epoch,
                }
                save_path = os.path.join(
                    self.log_dir,
                    f'DiT_epoch_{self.epoch:05d}_step_{self.iters:05d}.pth'
                )
                torch.save(state, save_path)
                print(f"Checkpoint saved at {save_path}")

            if self.iters >= self.max_steps:
                break

        print('Saving final model..')
        state = {
            'net': {key: self.model.module.models[key].state_dict() for key in self.model.module.models},
        }
        os.makedirs(self.log_dir, exist_ok=True)
        save_path = os.path.join(self.log_dir, 'ft_model.pth')
        torch.save(state, save_path)
        print(f"Final model saved at {save_path}")


def main(rank,args):
    # Setup distributed environment
    dist.init_process_group(backend="nccl", init_method="env://", world_size=args.n_gpus, rank=rank)
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    device = f'cuda:{rank}'
    trainer = Trainer(
        config_path=args.config,
        pretrained_ckpt_path=args.pretrained_ckpt,
        data_dir=args.dataset_dir,
        exp_dir=args.exp_dir,
        batch_size=args.batch_size,
        steps=args.max_steps,
        max_epochs=args.max_epochs,
        save_interval=args.save_every,
        num_workers=args.num_workers,
        device = device,
        local_rank=rank,
        n_gpu=args.n_gpus,
    )
    trainer.train(rank)
    dist.barrier()  # Ensure all processes complete before finishing
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/presets/config_dit_mel_seed_uvit_xlsr_tiny.yml')
    parser.add_argument('--pretrained-ckpt', type=str, default=None)
    parser.add_argument('--dataset-dir', type=str, default='/path/to/dataset')
    parser.add_argument('--exp-dir', type=str, default='exp')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--max-epochs', type=int, default=1000)
    parser.add_argument('--save-every', type=int, default=5004)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--n-gpus', type=int, default=5)
    
    args = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3,4,5"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(11451, 41919))
    #args.n_gpus = torch.cuda.device_count()
    main(0, args)
    #mp.spawn(main, nprocs=args.n_gpus, args=(args,))
