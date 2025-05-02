import torch.nn as nn

from lib.config.schema import ModelConfig
from .commons.common_layers import (
    NormalInitEmbedding as Embedding,
    XavierUniformInitLinear as Linear,
)
from .commons.tts_modules import LocalUpsample
from .decoder import DiffusionDecoder, ShallowDiffusionOutput
from .embedding import ParameterEmbeddings
from .encoder import LinguisticEncoder, MelodyEncoder
from .normalizer import FeatureNormalizer

__all__ = [
    "DiffSingerAcoustic",
    "DiffSingerVariance",
]


class DiffSingerAcoustic(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.linguistic_encoder = LinguisticEncoder(config=config.linguistic_encoder)
        self.local_upsample = LocalUpsample()  # tokens to frames
        self.use_spk_embed = config.use_spk_id
        if self.use_spk_embed:
            self.speaker_embedding = Embedding(config.num_spk, config.condition_dim)
        self.parameter_embeddings = ParameterEmbeddings(config=config.embeddings)
        self.spec_decoder = DiffusionDecoder(
            sample_dim=config.sample_dim,
            condition_dim=config.condition_dim,
            normalizer=FeatureNormalizer(
                num_channels=config.sample_dim, num_features=1, num_repeats=None,
                squeeze_channel_dim=False, squeeze_feature_dim=True,
                norm_mins=[config.normalization.spec_min],
                norm_maxs=[config.normalization.spec_max],
            ),
            config=config.spec_decoder
        )

    def forward(
            self, tokens, durations, languages, f0, spk_ids=None,
            spk_embed=None, spec_gt=None, infer=True, **kwargs
    ) -> ShallowDiffusionOutput:
        encoder_out = self.linguistic_encoder(tokens=tokens, durations=durations, languages=languages)
        cond, mask = self.local_upsample(encoder_out, ups=durations)
        if self.use_spk_embed:
            if spk_embed is None:
                spk_embed = self.speaker_embedding(spk_ids)[:, None, :]
            cond = cond + spk_embed
        cond = self.parameter_embeddings(cond, f0=f0, **kwargs)
        decoder_out = self.spec_decoder(condition=cond, sample_gt=spec_gt, infer=infer)
        return decoder_out, mask


class DiffSingerVariance(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.predict_pitch = config.prediction.predict_pitch
        var_norm_mins = []
        var_norm_maxs = []
        var_clip_mins = []
        var_clip_maxs = []
        variance_list = config.prediction.predicted_variance_names
        if config.prediction.predict_energy:
            var_norm_mins.append(config.normalization.energy_db_min)
            var_norm_maxs.append(config.normalization.energy_db_max)
            var_clip_mins.append(config.normalization.energy_db_min)
            var_clip_maxs.append(0.)
        if config.prediction.predict_breathiness:
            var_norm_mins.append(config.normalization.breathiness_db_min)
            var_norm_maxs.append(config.normalization.breathiness_db_max)
            var_clip_mins.append(config.normalization.breathiness_db_min)
            var_clip_maxs.append(0.)
        if config.prediction.predict_voicing:
            var_norm_mins.append(config.normalization.voicing_db_min)
            var_norm_maxs.append(config.normalization.voicing_db_max)
            var_clip_mins.append(config.normalization.voicing_db_min)
            var_clip_maxs.append(0.)
        if config.prediction.predict_tension:
            var_norm_mins.append(config.normalization.tension_logit_min)
            var_norm_maxs.append(config.normalization.tension_logit_max)
            var_clip_mins.append(config.normalization.tension_logit_min)
            var_clip_maxs.append(config.normalization.tension_logit_max)
        self.predict_variances = len(variance_list) > 0
        self.variance_list = variance_list
        if not self.predict_pitch and not self.predict_variances:
            raise ValueError("Nothing to predict.")

        self.linguistic_encoder = LinguisticEncoder(config=config.linguistic_encoder)
        self.local_upsample = LocalUpsample()
        self.use_spk_embed = config.use_spk_id
        if self.use_spk_embed:
            self.spk_embed = Embedding(config.num_spk, config.condition_dim)
        if self.predict_pitch:
            self.melody_encoder = MelodyEncoder(config=config.melody_encoder)
            self.pitch_predictor = DiffusionDecoder(
                sample_dim=config.normalization.pitch_repeat_bins,
                condition_dim=config.condition_dim,
                normalizer=FeatureNormalizer(
                    num_channels=1, num_features=1, num_repeats=config.normalization.pitch_repeat_bins,
                    squeeze_channel_dim=True, squeeze_feature_dim=True,
                    norm_mins=[config.normalization.pitd_norm_min],
                    norm_maxs=[config.normalization.pitd_norm_max],
                    clip_mins=[config.normalization.pitd_clip_min],
                    clip_maxs=[config.normalization.pitd_clip_max],
                ),
                config=config.pitch_predictor
            )
        if self.predict_variances:
            total_repeat_bins = config.normalization.variance_total_repeat_bins
            if total_repeat_bins % len(self.variance_list) != 0:
                raise ValueError(
                    f"variance_total_repeat_bins must be divisible by "
                    f"number of variances ({len(self.variance_list)})."
                )
            self.pitch_embedding = Linear(1, config.condition_dim)
            self.variance_predictor = DiffusionDecoder(
                sample_dim=total_repeat_bins,
                condition_dim=config.condition_dim,
                normalizer=FeatureNormalizer(
                    num_channels=1, num_features=len(self.variance_list),
                    num_repeats=total_repeat_bins // len(self.variance_list),
                    squeeze_channel_dim=True, squeeze_feature_dim=False,
                    norm_mins=var_norm_mins, norm_maxs=var_norm_maxs,
                    clip_mins=var_clip_mins, clip_maxs=var_clip_maxs,
                ),
                config=config.variance_predictor
            )

    def forward(
            self, tokens, durations, languages, spk_ids=None, spk_embed=None,
            note_midi=None, note_rest=None, note_dur=None, note_glide=None,
            base_pitch=None, pitch=None,
            infer=True, **kwargs
    ):
        linguistic_encoder_out = self.linguistic_encoder(tokens=tokens, durations=durations, languages=languages)
        cond, mask = self.local_upsample(linguistic_encoder_out, ups=durations)
        if self.use_spk_embed:
            if spk_embed is None:
                spk_embed = self.spk_embed(spk_ids)[:, None, :]
            cond = cond + spk_embed
        if self.predict_pitch:
            melody_encoder_out = self.melody_encoder(
                note_midi=note_midi, note_rest=note_rest, note_dur=note_dur, glide=note_glide
            )
            # TODO: add pitch retaking and expressiveness
            pitch_cond = cond + self.local_upsample(melody_encoder_out, ups=note_dur)[0]
            pitch_predictor_out = self.pitch_predictor(condition=pitch_cond, sample_gt=pitch - base_pitch, infer=infer)
            pitch_predictor_out = pitch_predictor_out.diff_out  # no shallow diffusion yet
            if infer:
                pitch_predictor_out = pitch_predictor_out + base_pitch
        else:
            pitch_predictor_out = None
        if self.predict_variances:
            variance_cond = cond + self.pitch_embedding(pitch[:, :, None])
            variance_predictor_out = self.variance_predictor(
                condition=variance_cond,
                sample_gt=[kwargs.get(v_name) for v_name in self.variance_list],
                infer=infer
            )
            variance_predictor_out = variance_predictor_out.diff_out  # no shallow diffusion yet
        else:
            variance_predictor_out = None
        return pitch_predictor_out, variance_predictor_out, mask
