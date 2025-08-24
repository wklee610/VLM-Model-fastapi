from app.manager.model_manager import ModelContainer
from app.model.schemas import Conversation

class ChatGenerator:
    def __init__(
        self, 
        llava_container: ModelContainer
    ) -> None:
        self.model = llava_container.model
        self.processor = llava_container.processor
        self.device = llava_container.device
        self.torch_dtype = llava_container.torch_dtype
        self.tensor_type = llava_container.tensor_type
        self.max_new_tokens = llava_container.max_new_tokens
        self.system_prompt = llava_container.system_prompt
        self.few_shots = llava_container.few_shots

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