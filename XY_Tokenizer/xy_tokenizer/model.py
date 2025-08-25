# -*- coding: utf-8 -*-
import yaml
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


from .nn.feature_extractor import MelFeatureExtractor
from .nn.modules import OmniAudioEncoder, OmniAudioDecoder, ResidualDownConv, UpConv, Transformer, Vocos
from .nn.quantizer import ResidualVQ

class XY_Tokenizer(nn.Module):
    def __init__(self, generator_params):
        super().__init__()
        # Basic parameters
        self.input_sample_rate = generator_params['input_sample_rate']
        self.output_sample_rate = generator_params['output_sample_rate']
        
        self.encoder_downsample_rate = generator_params['encoder_downsample_rate']
        self.decoder_upsample_rate = generator_params['decoder_upsample_rate']
        self.code_dim = generator_params['quantizer_kwargs']['input_dim']
        
        ## Codec part

        ## Semantic channel
        self.semantic_encoder = OmniAudioEncoder(**generator_params['semantic_encoder_kwargs'])
        
        self.semantic_encoder_adapter = Transformer(**generator_params['semantic_encoder_adapter_kwargs'])
        
        ## Acoustic channel
        self.acoustic_encoder = OmniAudioEncoder(**generator_params['acoustic_encoder_kwargs'])
        
        ## Semantic & acoustic shared parameters
        self.pre_rvq_adapter = Transformer(**generator_params['pre_rvq_adapter_kwargs'])
        
        self.downsample = ResidualDownConv(**generator_params['downsample_kwargs'])
        
        self.quantizer = ResidualVQ(**generator_params['quantizer_kwargs'])
        self.nq = generator_params['quantizer_kwargs']['num_quantizers']

        self.post_rvq_adapter = Transformer(**generator_params['post_rvq_adapter_kwargs'])
                    
        ## Acoustic channel
        self.upsample = UpConv(**generator_params['upsample_kwargs'])

        self.acoustic_decoder = OmniAudioDecoder(**generator_params['acoustic_decoder_kwargs'])

        self.enhanced_vocos = Vocos(**generator_params['vocos_kwargs'])

        ## Feature extractor
        self.feature_extractor = MelFeatureExtractor(**generator_params['feature_extractor_kwargs'])

    @torch.inference_mode()
    def inference_tokenize(self, x, input_lengths):
        """
            Input:
                x: Waveform tensor # (B, 1, T), T <= 30s * sample_rate
                input_lengths: Valid length for each sample # (B,)
            Output:
                dict: Contains the following key-value pairs
                    "zq": Quantized embeddings # (B, D, T)
                    "codes": Quantization codes # (nq, B, T)
                    "codes_lengths": Quantization code lengths # (B,)
        """
        list_x = [xi[:, :x_len].reshape(-1).cpu().numpy() for xi, x_len in zip(x, input_lengths)]
        features = self.feature_extractor(
            list_x,
            sampling_rate=self.input_sample_rate,
            return_tensors="pt",
            return_attention_mask=True
        )
        input_mel = features['input_features'].to(x.device).to(x.dtype) # (B, D, 3000)
        audio_attention_mask = features['attention_mask'].to(x.device) # (B, 3000)
        
        # Get batch size and sequence length of the input
        mel_output_length = torch.sum(audio_attention_mask, dim=-1).long() # (B,)
        
        # Semantic channel
        semantic_encoder_output, semantic_encoder_output_length = self.semantic_encoder(input_mel, mel_output_length) # (B, D, T), 100hz -> 50hz
        
        semantic_encoder_adapter_output, semantic_encoder_adapter_output_length = self.semantic_encoder_adapter(semantic_encoder_output, semantic_encoder_output_length) # (B, D, T), 50hz
        
        # Acoustic channel
        acoustic_encoder_output, acoustic_encoder_output_length = self.acoustic_encoder(input_mel, mel_output_length) # (B, D, T), 100hz -> 50hz
        
        # Semantic & acoustic mixing
        concated_semantic_acoustic_channel = torch.concat([semantic_encoder_adapter_output, acoustic_encoder_output], dim=1) # (B, D, T)
        concated_semantic_acoustic_channel_length = acoustic_encoder_output_length
        
        pre_rvq_adapter_output, pre_rvq_adapter_output_length = self.pre_rvq_adapter(concated_semantic_acoustic_channel, concated_semantic_acoustic_channel_length) # (B, D, T), 50hz
        
        downsample_output, downsample_output_length = self.downsample(pre_rvq_adapter_output, pre_rvq_adapter_output_length) # (B, D, T), 50hz -> 12.5hz

        zq, codes, vq_loss, _, quantizer_output_length = self.quantizer(downsample_output, downsample_output_length) # (B, D, T), (nq, B, T), (nq,), (nq, B, D, T), (B,)

        return {
            "zq": zq, # (B, D, T)
            "codes": codes, # (nq, B, T)
            "codes_lengths": quantizer_output_length # (B,)
        }
      
    @torch.inference_mode()  
    def inference_detokenize(self, codes, codes_lengths):
        """
            Input:
                codes: Quantization codes # (nq, B, T)
                codes_lengths: Quantization code lengths for each sample # (B,)
            Output:
                dict: Contains the following key-value pairs
                    "y": Synthesized audio waveform # (B, 1, T)
                    "output_length": Output lengths # (B,)
        """
        zq = self.quantizer.decode_codes(codes) # (B, D, T)
        
        post_rvq_adapter_output, post_rvq_adapter_output_length = self.post_rvq_adapter(zq, codes_lengths) # (B, D, T), 12.5hz
        
        # Acoustic channel            
        upsample_output, upsample_output_length = self.upsample(post_rvq_adapter_output, post_rvq_adapter_output_length) # (B, D, T), 12.5hz -> 50hz

        acoustic_decoder_output, acoustic_decoder_output_length = self.acoustic_decoder(upsample_output, upsample_output_length) # (B, D, T), 50hz -> 100hz

        y, vocos_output_length = self.enhanced_vocos(acoustic_decoder_output, acoustic_decoder_output_length) # (B, 1, T), 100hz -> 16khz
        
        return {
            "y": y, # (B, 1, T)
            "output_length": vocos_output_length, # (B,)
        }
        
    @torch.inference_mode()
    def encode(self, wav_list, overlap_seconds=10, device=torch.device("cuda")):
        """
            Input:
                wav_list: List of audio waveforms, each with potentially different length, may exceed 30 seconds # B * (T,)
                overlap_seconds: Overlap in seconds, process 30 seconds at a time, keeping (30 - overlap_seconds) seconds of valid output
            Output:
                dict: Contains the following key-value pairs
                    "codes_list": List of quantization codes # B * (nq, T)
        """
        duration_seconds = 30 - overlap_seconds
        chunk_size = int(30 * self.input_sample_rate) # Maximum samples per chunk
        duration_size = int(duration_seconds * self.input_sample_rate) # Valid output samples per chunk
        code_duration_length = duration_size // self.encoder_downsample_rate # Valid code length per chunk

        # Get maximum waveform length
        max_length = max(len(wav) for wav in wav_list)
        batch_size = len(wav_list)
        wav_tensor = torch.zeros(batch_size, 1, max_length, device=device)
        input_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        for i, wav in enumerate(wav_list):
            wav_tensor[i, 0, :len(wav)] = wav
            input_lengths[i] = len(wav) # (B,)

        # Calculate number of chunks needed
        max_chunks = (max_length + duration_size - 1) // duration_size
        codes_list = []

        # Process the entire batch in chunks
        for chunk_idx in range(max_chunks):
            start = chunk_idx * duration_size
            end = min(start + chunk_size, max_length)
            chunk = wav_tensor[:, :, start:end] # (B, 1, T')
            chunk_lengths = torch.clamp(input_lengths - start, 0, end - start) # (B,)

            # Skip empty chunks
            if chunk_lengths.max() == 0:
                continue

            # Encode
            result = self.inference_tokenize(chunk, chunk_lengths) # {"zq": (B, D, T'), "codes": (nq, B, T'), "codes_lengths": (B,)}
            chunk_codes = result["codes"] # (nq, B, T')
            chunk_code_lengths = result["codes_lengths"] # (B,)

            # Extract valid portion
            valid_code_lengths = torch.clamp(chunk_code_lengths, 0, code_duration_length) # (B,)
            valid_chunk_codes = torch.zeros(self.nq, batch_size, code_duration_length, device=device, dtype=chunk_codes.dtype)
            for b in range(batch_size):
                if valid_code_lengths[b] > 0:
                    valid_chunk_codes[:, b, :valid_code_lengths[b]] = chunk_codes[:, b, :valid_code_lengths[b]] # (nq, B, valid_code_length)

            codes_list.append(valid_chunk_codes) # (nq, B, valid_code_length)

        # Concatenate all chunks
        if codes_list:
            codes_tensor = torch.cat(codes_list, dim=-1) # (nq, B, T_total)
            codes_list = [codes_tensor[:, i, :input_lengths[i] // self.encoder_downsample_rate] for i in range(batch_size)] # B * (nq, T)
        else:
            codes_list = [torch.zeros(self.nq, 0, device=device, dtype=torch.long) for _ in range(batch_size)] # B * (nq, 0)

        return {
            "codes_list": codes_list # B * (nq, T)
        }
        
    @torch.inference_mode()
    def decode(self, codes_list, overlap_seconds=10, device=torch.device("cuda")):
        """
            Input:
                codes_list: List of quantization codes # B * (nq, T)
                overlap_seconds: Overlap in seconds, process 30 seconds at a time, keeping (30 - overlap_seconds) seconds of valid output
            Output:
                dict: Contains the following key-value pairs
                    "syn_wav_list": List of synthesized audio waveforms # B * (T,)
        """
        duration_seconds = 30 - overlap_seconds
        chunk_code_length = int(30 * self.input_sample_rate // self.encoder_downsample_rate) # Maximum code length per chunk
        duration_code_length = int(duration_seconds * self.input_sample_rate // self.encoder_downsample_rate) # Valid code length per chunk
        duration_wav_length = duration_code_length * self.decoder_upsample_rate # Valid waveform length per chunk

        # Get maximum code length
        max_code_length = max(codes.shape[-1] for codes in codes_list)
        batch_size = len(codes_list)
        codes_tensor = torch.zeros(self.nq, batch_size, max_code_length, device=device, dtype=torch.long)
        code_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        for i, codes in enumerate(codes_list):
            codes_tensor[:, i, :codes.shape[-1]] = codes.to(device)
            code_lengths[i] = codes.shape[-1] # (B,)

        # Calculate number of chunks needed
        max_chunks = (max_code_length + duration_code_length - 1) // duration_code_length
        wav_list = []

        # Process the entire batch in chunks
        for chunk_idx in range(max_chunks):
            start = chunk_idx * duration_code_length
            end = min(start + chunk_code_length, max_code_length)
            chunk_codes = codes_tensor[:, :, start:end] # (nq, B, T')
            chunk_code_lengths = torch.clamp(code_lengths - start, 0, end - start) # (B,)

            # Skip empty chunks
            if chunk_code_lengths.max() == 0:
                continue

            # Decode
            result = self.inference_detokenize(chunk_codes, chunk_code_lengths) # {"y": (B, 1, T'), "output_length": (B,)}
            chunk_wav = result["y"] # (B, 1, T')
            chunk_wav_lengths = result["output_length"] # (B,)

            # Extract valid portion
            valid_wav_lengths = torch.clamp(chunk_wav_lengths, 0, duration_wav_length) # (B,)
            valid_chunk_wav = torch.zeros(batch_size, 1, duration_wav_length, device=device)
            for b in range(batch_size):
                if valid_wav_lengths[b] > 0:
                    valid_chunk_wav[b, :, :valid_wav_lengths[b]] = chunk_wav[b, :, :valid_wav_lengths[b]] # (B, 1, valid_wav_length)

            wav_list.append(valid_chunk_wav) # (B, 1, valid_wav_length)

        # Concatenate all chunks
        if wav_list:
            wav_tensor = torch.cat(wav_list, dim=-1) # (B, 1, T_total)
            syn_wav_list = [wav_tensor[i, 0, :code_lengths[i] * self.decoder_upsample_rate] for i in range(batch_size)] # B * (T,)
        else:
            syn_wav_list = [torch.zeros(0, device=device) for _ in range(batch_size)] # B * (0,)
            
        return {
            "syn_wav_list": syn_wav_list # B * (T,)
        }
    
    @classmethod
    def load_from_checkpoint(cls, config_path: str, ckpt_path: str):
        # Load model from configuration file and checkpoint
        logging.info(f"Loading model from {config_path} and {ckpt_path}")
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create model instance
        model = cls(config['generator_params'])
        
        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        # Check if checkpoint contains 'generator' key
        if 'generator' in checkpoint:
            model.load_state_dict(checkpoint['generator'])
        else:
            model.load_state_dict(checkpoint)
        
        return model