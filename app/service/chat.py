from vllm import (
    LLM, 
    SamplingParams
)
from app.manager.model_manager import ModelContainer
from app.model.schemas import Conversation

class ChatGenerator:
    def __init__(
        self, 
        model_container: ModelContainer
    ) -> None:
        self.model = model_container.model
        self.processor = model_container.processor
        self.device = model_container.device
        self.torch_dtype = model_container.torch_dtype
        self.tensor_type = model_container.tensor_type
        self.max_new_tokens = model_container.max_new_tokens
        self.system_prompt = model_container.system_prompt
        self.few_shots = model_container.few_shots
        self.use_vllm = isinstance(self.model, LLM)
        self.sampling_params = model_container.sampling_params

    def generate(
        self, 
        conversation: Conversation
    ) -> str:
        try:
            conversation_dict = [msg.model_dump() for msg in conversation.root]
            full_prompt = []
            if self.system_prompt:
                full_prompt.append(
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": self.system_prompt
                            }
                        ]
                    }
                )
            if self.few_shots:
                full_prompt.extend(self.few_shots)
            full_prompt.extend(conversation_dict)

            if self.use_vllm:
                full_prompt = self.processor.apply_chat_template(
                    full_prompt,
                    add_generation_prompt=True,
                    tokenize=False,
                    return_dict=False
                )
                
                outputs = self.model.generate([full_prompt], sampling_params=self.sampling_params)

                output_text = outputs[0].outputs[0].text
                return output_text
            else:
                inputs = self.processor.apply_chat_template(
                    full_prompt,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors=self.tensor_type
                ).to(self.device, self.torch_dtype)

                generate_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=self.max_new_tokens
                )
        
                generate_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generate_ids)
                ]
                output = self.processor.decode(generate_ids_trimmed[0], skip_special_tokens=True)
                return output

        except Exception as e:
            print(f"Error during model generation: {e}")
            return e