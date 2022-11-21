import os, sys, pdb
import logging
import numpy as np

import wandb

# from tensorboardX import SummaryWriter
from torch.optim.optimizer import Optimizer, required
from transformers import AdamW, get_linear_schedule_with_warmup


class ExperienceLogger(object):
    def __init__(self, args, checkpoint_dir):
        self.args = args
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.FileHandler(args.output_dir + "/exp.log"))

        self.epoch = 0
        self.global_step = 0
        self.eval_step = 0
        self.log_interval = args.log_interval

        self.best_score = float("-inf")
        self.task = args.task
        self.mtype = args.model_type
        self.verbose = args.verbose

        self.output_dir = args.output_dir
        self.filepath = os.path.join(checkpoint_dir, "pytorch_model.pt")

    def start_train(self, num_examples, total_step):
        self.logger.info("***** Running training *****")
        self.logger.info(
            f"  Train examples: {num_examples}, Batch size: {self.args.batch_size}"
        )
        self.logger.info(
            f"  Num epochs: {self.args.epochs}, Optimization steps: {total_step}"
        )
        self.logger.info(f"  Running experiment for {self.task} {self.args.filename}")

    def start_eval(self, num_examples, kind):
        self.epoch += 1
        self.eval_loss = 0
        self.batch_steps = 0

        if self.verbose:
            epoch_msg = f"epoch {self.epoch} evaluation"
            self.logger.info(f"***** Running {epoch_msg} for {kind} {self.mtype} *****")
            self.logger.info(f"  Num evaluation examples: {num_examples}")

    def end_eval(self, result, kind):
        self.logger.info("***** Eval results for {} *****".format(kind))
        for key in sorted(result.keys()):
            self.logger.info("  %s = %s", key, str(result[key]))

    def log_train(self, step, loss, result, metric):
        if self.log_interval > 0 and self.global_step % self.log_interval == 0:
            log_str = "Step {:>6d} | Loss {:5.4f}".format(step, loss)
            self.add_scalar("train", "loss", loss, self.global_step)

            if self.verbose:
                value = round(result[metric], 3)
                log_str += f" | {metric} {value}"
                self.add_scalar("train", metric.lower(), value, self.global_step)
            self.logger.info(log_str)

        self.global_step += 1

    def log_dev(self, step, metric, value):
        self.eval_step += 1

        avg_eval_loss = round(self.eval_loss / self.batch_steps, 4)
        log_str = "Eval {:3d} | Loss {} | {} {}".format(
            step, avg_eval_loss, metric, value
        )
        self.logger.info(log_str)
        self.add_scalar("dev", "loss", avg_eval_loss, self.global_step)
        self.add_scalar("dev", metric.lower(), value, self.global_step)

    def init_tb_writers(self):
        raise NotImplementedError
        # self.train_writer = SummaryWriter(log_dir=self.output_dir + "/train")
        # self.dev_writer = SummaryWriter(log_dir=self.output_dir + "/dev")

    def add_scalar(self, mode, name, value, step):
        wandb.log(
            {
                f"{mode}-{name}": value,
            },
            step=step,
        )
        # if mode == "train":
        # self.train_writer.add_scalar(name, value, step)
        # elif mode == "dev":
        # self.dev_writer.add_scalar(name, value, step)


# TODO: AdamW
class RAdam(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        degenerated_to_sgd=True,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if (
            isinstance(params, (list, tuple))
            and len(params) > 0
            and isinstance(params[0], dict)
        ):
            for param in params:
                if "betas" in param and (
                    param["betas"][0] != betas[0] or param["betas"][1] != betas[1]
                ):
                    param["buffer"] = [[None, None, None] for _ in range(10)]
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            buffer=[[None, None, None] for _ in range(10)],
        )
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError("RAdam does not support sparse gradients")

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].type_as(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state["step"] += 1
                buffered = group["buffer"][int(state["step"] % 10)]
                if state["step"] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state["step"]
                    beta2_t = beta2 ** state["step"]
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state["step"] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t)
                            * (N_sma - 4)
                            / (N_sma_max - 4)
                            * (N_sma - 2)
                            / N_sma
                            * N_sma_max
                            / (N_sma_max - 2)
                        ) / (1 - beta1 ** state["step"])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state["step"])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group["weight_decay"] != 0:
                        p_data_fp32.add_(
                            -group["weight_decay"] * group["lr"], p_data_fp32
                        )
                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                    p_data_fp32.addcdiv_(-step_size * group["lr"], exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group["weight_decay"] != 0:
                        p_data_fp32.add_(
                            -group["weight_decay"] * group["lr"], p_data_fp32
                        )
                    p_data_fp32.add_(-step_size * group["lr"], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss
