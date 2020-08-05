import torch.nn.functional as F


def standard_discriminator_loss(real_logits, fake_logits):
    real_loss = -F.logsigmoid(real_logits)
    fake_loss = fake_logits - F.logsigmoid(fake_logits)
    return real_loss, fake_loss


def standard_generator_loss(fake_logits):
    return -fake_logits + F.logsigmoid(fake_logits)


def logD_discriminator_loss(real_logits, fake_logits):
    real_loss = -0.5 * F.logsigmoid(real_logits)
    fake_loss = 0.5 * (fake_logits - F.logsigmoid(fake_logits))
    return real_loss, fake_loss


def logD_generator_loss(fake_logits):
    return F.softplus(-fake_logits)


def get_loss_fn(loss_fn):
    if loss_fn == 'standard':
        return standard_discriminator_loss, standard_generator_loss
    elif loss_fn == 'logD':
        return logD_discriminator_loss, logD_generator_loss
    else:
        raise ValueError("Invalid loss")
