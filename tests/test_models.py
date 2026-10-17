"""Unit tests for forecasting models."""

import pytest
import torch

from src.models.linear import LinearBaseline
from src.models.encoder_only import EncoderOnlyTransformer
from src.models.encoder_decoder import EncoderDecoderTransformer
from src.models.deepar import DeepAR
from src.models.temporal_fusion_transformer import TemporalFusionTransformer


BATCH = 4
INPUT_WINDOW = 28
HORIZON = 28
N_FEATURES = 15


@pytest.fixture
def sample_input():
    """Generate sample input tensors."""
    x_sales = torch.randn(BATCH, INPUT_WINDOW)
    x_features = torch.randn(BATCH, INPUT_WINDOW, N_FEATURES)
    return x_sales, x_features


class TestLinearBaseline:
    def test_output_shape(self, sample_input):
        x_sales, x_features = sample_input
        model = LinearBaseline(INPUT_WINDOW, HORIZON, n_features=N_FEATURES + 1)
        out = model(x_sales, x_features)
        assert out.shape == (BATCH, HORIZON)

    def test_without_features(self):
        x_sales = torch.randn(BATCH, INPUT_WINDOW)
        model = LinearBaseline(INPUT_WINDOW, HORIZON, n_features=1)
        out = model(x_sales)
        assert out.shape == (BATCH, HORIZON)

    def test_gradient_flow(self, sample_input):
        x_sales, x_features = sample_input
        model = LinearBaseline(INPUT_WINDOW, HORIZON, n_features=N_FEATURES + 1)
        out = model(x_sales, x_features)
        loss = out.sum()
        loss.backward()
        assert all(p.grad is not None for p in model.parameters())


class TestEncoderOnlyTransformer:
    def test_output_shape(self, sample_input):
        x_sales, x_features = sample_input
        model = EncoderOnlyTransformer(
            n_features=N_FEATURES + 1,
            d_model=64, n_heads=4, n_layers=2, d_ff=128,
            input_window=INPUT_WINDOW, forecast_horizon=HORIZON,
        )
        out = model(x_sales, x_features)
        assert out.shape == (BATCH, HORIZON)

    def test_different_batch_sizes(self, sample_input):
        model = EncoderOnlyTransformer(
            n_features=N_FEATURES + 1,
            d_model=64, n_heads=4, n_layers=2, d_ff=128,
            input_window=INPUT_WINDOW, forecast_horizon=HORIZON,
        )
        for bs in [1, 8, 16]:
            x = torch.randn(bs, INPUT_WINDOW)
            f = torch.randn(bs, INPUT_WINDOW, N_FEATURES)
            assert model(x, f).shape == (bs, HORIZON)


class TestEncoderDecoderTransformer:
    def test_output_shape_teacher_forcing(self, sample_input):
        x_sales, x_features = sample_input
        y = torch.randn(BATCH, HORIZON)
        model = EncoderDecoderTransformer(
            n_features=N_FEATURES + 1,
            d_model=64, n_heads=4,
            n_encoder_layers=2, n_decoder_layers=2, d_ff=128,
            input_window=INPUT_WINDOW, forecast_horizon=HORIZON,
        )
        out = model(x_sales, x_features, y=y)
        assert out.shape == (BATCH, HORIZON)

    def test_output_shape_autoregressive(self, sample_input):
        x_sales, x_features = sample_input
        model = EncoderDecoderTransformer(
            n_features=N_FEATURES + 1,
            d_model=64, n_heads=4,
            n_encoder_layers=2, n_decoder_layers=2, d_ff=128,
            input_window=INPUT_WINDOW, forecast_horizon=HORIZON,
        )
        out = model(x_sales, x_features)
        assert out.shape == (BATCH, HORIZON)

    def test_gradient_flow(self, sample_input):
        x_sales, x_features = sample_input
        y = torch.randn(BATCH, HORIZON)
        model = EncoderDecoderTransformer(
            n_features=N_FEATURES + 1,
            d_model=64, n_heads=4,
            n_encoder_layers=2, n_decoder_layers=2, d_ff=128,
            input_window=INPUT_WINDOW, forecast_horizon=HORIZON,
        )
        out = model(x_sales, x_features, y=y)
        out.sum().backward()
        assert all(
            p.grad is not None for p in model.parameters() if p.requires_grad
        )


class TestDeepAR:
    def test_distribution_and_output_shapes(self, sample_input):
        x_sales, x_features = sample_input
        model = DeepAR(
            n_features=N_FEATURES + 1,
            hidden_size=32,
            n_layers=1,
            input_window=INPUT_WINDOW,
            forecast_horizon=HORIZON,
        )
        mean, scale = model.distribution(x_sales, x_features)
        assert mean.shape == scale.shape == (BATCH, HORIZON)
        assert torch.all(scale > 0)

    def test_likelihood_gradient_flow(self, sample_input):
        x_sales, x_features = sample_input
        y = torch.rand(BATCH, HORIZON)
        model = DeepAR(
            n_features=N_FEATURES + 1,
            hidden_size=32,
            n_layers=1,
            input_window=INPUT_WINDOW,
            forecast_horizon=HORIZON,
        )
        model.training_loss(x_sales, x_features, y).backward()
        assert all(p.grad is not None for p in model.parameters())


class TestTemporalFusionTransformer:
    def test_output_shape_and_gradient_flow(self, sample_input):
        x_sales, x_features = sample_input
        model = TemporalFusionTransformer(
            n_features=N_FEATURES + 1,
            d_model=32,
            n_heads=4,
            d_ff=64,
            input_window=INPUT_WINDOW,
            forecast_horizon=HORIZON,
        )
        output = model(x_sales, x_features)
        assert output.shape == (BATCH, HORIZON)
        output.sum().backward()
        assert all(p.grad is not None for p in model.parameters())
