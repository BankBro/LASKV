from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Step 1: Load pre-trained GPT-2 model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Step 2: Prepare input prompt
prompt = ("Write a chapter from the fantasy adventure novel \"The City of Eternal Mist.\" "
            "The story is set in a distant past where technology and magic coexist, "
            "in an ancient metropolis shrouded in perpetual fog. "
            "The city, known as the City of Eternal Mist, is surrounded by towering stone walls, "
            "with winding streets and ancient buildings inside. At its heart stands a soaring tower, "
            "said to contain the secrets of the city. The inhabitants worship the elements of nature "
            "and believe that all things have a spirit. The main characters are Alison, a young and brave "
            "warrior with the ability to control the wind; Sebastian, a mysterious mage skilled in the use of fire; "
            "Melina, a clever and agile thief adept at stealth and lock-picking; and Oliver, "
            "a wise old scholar who knows the history of the City of Eternal Mist like the back of his hand. "
            "The chapter should focus on the team's journey through the city as they navigate its treacherous "
            "streets and ancient structures. Alison discovers an ancient book during one of her expeditions, "
            "which speaks of a mystical treasure hidden deep within the City of Eternal Mist. "
            "She invites Sebastian, Melina, and Oliver to join her on a quest to find the treasure. "
            "As they journey through the city, they encounter various challenges, including traps, puzzles, "
            "and mythical creatures guarding the treasure. They also face internal conflicts as their motivations "
            "and loyalties are tested. Include vivid descriptions of the environment, the characters' interactions, "
            "and the challenges they face. Highlight the tension and suspense as they get closer to the treasure, "
            "and explore the dynamics between the characters as they confront both external and internal conflicts. "
            "End the chapter with a cliffhanger that leaves the reader eager to know what happens next.")
inputs = tokenizer(prompt, return_tensors='pt')

# Step 3: Generate output and get KV cache
with torch.no_grad():
    outputs = model(**inputs, use_cache=True)
    logits, past_key_values = outputs.logits, outputs.past_key_values

# Step 4: Decode and print the generated text
generated_ids = torch.argmax(logits, dim=-1)
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("Generated Text:")
print(generated_text)

# Step 5: Save KV cache to file
torch.save(past_key_values, 'kv_cache.pt')
print("KV cache has been saved to 'kv_cache.pt'")

# Step 6: Read KV cache from file
loaded_past_key_values = torch.load('kv_cache.pt')
print("\nLoaded KV Cache from 'kv_cache.pt'")
# print(loaded_past_key_values.shape())



