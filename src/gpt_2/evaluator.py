import torch
import wandb
import os


class Evaluators:
    def __init__(self, model, eval_dataloader, device, master_process, ddp):
        self.model = model
        self.eval_dataloader = eval_dataloader
        self.device = device
        self.master_process = master_process
        self.ddp = ddp

    def estimate_validation_loss(self, step, checkpoint_model=False, max_steps=None):
        """
        Estimate average loss on both training and validation sets.
        This provides a more stable estimate than single-batch loss.

        Returns:
            dict: Contains 'train' and 'val' average losses
        """
        self.model.eval()
        val_loss_accumulator = torch.tensor(0.0, device=self.device)
        self.eval_dataloader.reset()
        val_loss_steps = 20

        with torch.no_grad():
            for k in range(val_loss_steps):
                X, Y = self.val_dataloader.next_batch()
                X = X.to(self.device)
                Y = Y.to(self.device)
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    _, loss = self.model(X, Y)
                loss = loss / val_loss_steps
                val_loss_accumulator += loss

        self.model.train()

        if self.ddp:
            torch.distributed.all_reduce(
                val_loss_accumulator, op=torch.distributed.ReduceOp.AVG
            )

        if self.master_process:
            print(f"Step {step} | Validation loss: {val_loss_accumulator.item():.4f}")
            wandb.log({"val_loss": val_loss_accumulator.item()}, step=step)

        if checkpoint_model or step == max_steps - 1:
            checkpoint = {
                "model": self.model.state_dict(),
                "config": self.model.config,
                "step": step,
                "val_loss": val_loss_accumulator.item(),
            }
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            torch.save(
                checkpoint, f"{parent_dir}/checkpoints/gpt2_checkpoint_{step}.pt"
            )
