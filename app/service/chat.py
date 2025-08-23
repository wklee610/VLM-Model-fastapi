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

    def generate(
        self, 
        conversation: Conversation
    ) -> str:
        try:
            conversation_dict = [msg.model_dump() for msg in conversation.root]
            inputs = self.processor.apply_chat_template(
                conversation_dict,
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