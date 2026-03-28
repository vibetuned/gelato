import torch
from transformers import Trainer
from transformers.generation.logits_process import LogitsProcessorList

class GelatoSTATICTrainer(Trainer):
    def __init__(self, *args, static_processor=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize the STATIC Bouncer
        self.static_processor = static_processor
        
        # Wrap it in the Hugging Face LogitsProcessorList
        self.logits_processor_list = LogitsProcessorList([static_processor]) if static_processor else None

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Overrides the evaluation step to perform Constrained Generation.
        """
        # 1. Standard Validation Loss Computation (During training steps)
        if prediction_loss_only:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            # Compute standard cross-entropy loss
            with self.compute_loss_context_manager():
                outputs = model(**inputs)
                loss = outputs.loss.detach() if outputs.loss is not None else None

            # 2. STATIC Constrained Generation
            # Call the custom generate method we added to GelatoModel
            generated_tokens = model.generate(
                pixel_values=inputs["pixel_values"],
                logits_processor=self.logits_processor_list,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            # Grab the ground truth labels to return for metric calculation
            labels = inputs["labels"]

        # Hugging Face expects padded tensors of equal length to calculate metrics
        # We pad the generated sequences to match the longest sequence in the batch
        if generated_tokens.shape[1] < labels.shape[1]:
            padding_length = labels.shape[1] - generated_tokens.shape[1]
            pad_tensor = torch.full(
                (generated_tokens.shape[0], padding_length), 
                self.tokenizer.pad_token_id, 
                dtype=generated_tokens.dtype, 
                device=generated_tokens.device
            )
            generated_tokens = torch.cat([generated_tokens, pad_tensor], dim=-1)
        elif generated_tokens.shape[1] > labels.shape[1]:
             generated_tokens = generated_tokens[:, :labels.shape[1]]

        return (loss, generated_tokens, labels)