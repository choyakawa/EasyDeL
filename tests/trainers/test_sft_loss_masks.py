import jax.numpy as jnp

from easydel.infra.loss_utils import ForCausalLMLoss, LossConfig


def test_assistant_masks_override_attention_mask_for_causal_lm_loss():
    logits = jnp.array(
        [
            [
                [8.0, 0.0, 0.0],
                [0.0, 8.0, 0.0],
                [0.0, 0.0, 8.0],
                [8.0, 0.0, 0.0],
            ]
        ],
        dtype=jnp.float32,
    )
    labels = jnp.array([[0, 1, 2, 0]], dtype=jnp.int32)
    attention_mask = jnp.array([[1, 1, 1, 1]], dtype=jnp.int32)
    assistant_masks = jnp.array([[0, 0, 1, 0]], dtype=jnp.int32)

    metrics = ForCausalLMLoss(
        logits=logits,
        labels=labels,
        attention_mask=attention_mask,
        assistant_masks=assistant_masks,
        config=LossConfig(),
    )

    assert float(metrics.weight_sum) == 1.0


def test_completion_mask_is_used_when_assistant_mask_is_absent():
    logits = jnp.array(
        [
            [
                [8.0, 0.0, 0.0],
                [0.0, 8.0, 0.0],
                [0.0, 0.0, 8.0],
                [8.0, 0.0, 0.0],
            ]
        ],
        dtype=jnp.float32,
    )
    labels = jnp.array([[0, 1, 2, 0]], dtype=jnp.int32)
    attention_mask = jnp.array([[1, 1, 1, 1]], dtype=jnp.int32)
    completion_mask = jnp.array([[0, 1, 1, 0]], dtype=jnp.int32)

    metrics = ForCausalLMLoss(
        logits=logits,
        labels=labels,
        attention_mask=attention_mask,
        completion_mask=completion_mask,
        config=LossConfig(),
    )

    assert float(metrics.weight_sum) == 2.0
