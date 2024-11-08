"""
You may modify the signal generation pipeline as you wish.

We use an LLM to generate a sentiment score according to the prompt below. 

You can improve the sentiment analysis here or generate your own signal.
"""

import re
import torch

SAMPLE_PROMPT = """
Task: Analyze the following news headline about a stock and provide a sentiment score between -{signal_strengh} and {signal_strengh}, where:
- -{signal_strengh} represents a highly negative sentiment, likely indicating a substantial decline in stock performance.
- -{threshold} represents a moderate negative sentiment, suggesting a slight potential decline in stock performance.
- 0 represents neutral sentiment, indicating no significant impact on stock performance.
- {threshold} represents a moderate positive sentiment, indicating potential for slight stock growth.
- {signal_strengh} represents a highly positive sentiment, indicating significant potential for stock appreciation.

Consider the likely influence of market feedback from previous price movements and sentiment trends:
- How has the stock's price responded to similar news in the past?
- Does the headline align with prevailing market sentiment, or does it contradict current trends?
- How might this sentiment lead to a change in the stock's behavior, considering both historical price patterns and market expectations?

Do not provide any explanations or reasoning. Output only a single integer in the range of -{signal_strengh} to {signal_strengh} based on the sentiment of the news and its potential impact on stock performance.

News headline: "{news}"

Price Data: "{prices}"

Generate only a single integer value for the sentiment score after the colon. 

Single Integer Sentiment score:
"""

def _generate_signal(tokenizer, model, device, news, prices, signal_strengh, threshold):
    """Using model forward pass to do backprop"""
    prompt = SAMPLE_PROMPT.format(signal_strengh=signal_strengh, threshold=threshold, news=news, prices=prices)
    inputs = tokenizer(prompt, return_tensors="pt")  # .to(device)

    generated_ids = inputs["input_ids"]
    log_probs = []
    max_new_tokens = 1

    for _ in range(max_new_tokens):
        outputs = model(input_ids=generated_ids)
        logits = outputs.logits  # shape: [batch_size, seq_length, vocab_size]

        next_token_logits = logits[:, -1, :]
        next_token_probs = torch.softmax(next_token_logits, dim=-1)

        next_token_id = torch.multinomial(next_token_probs, num_samples=1)

        token_log_prob = torch.log(next_token_probs[0, next_token_id[0, 0]])
        log_probs.append(token_log_prob)

        generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

    total_log_prob = torch.stack(log_probs).sum()

    output_string = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

    match = re.search(r"Sentiment score:\s*(-?\d+(?:\.\d+)?)", output_string)
    signal_strengh = float(match.group(1)) if match else 0

    return signal_strengh, total_log_prob


def _generate_eval_signal(tokenizer, model, device, news, prices, signal_strengh, threshold):
    prompt = SAMPLE_PROMPT.format(signal_strengh=signal_strengh, threshold=threshold, news=news, prices=prices)

    # using news signals, prompt model for a scaled sentiment scorea
    input = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**input, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
    output_string = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print(output_string)
    match = re.search(r"Sentiment score:\s*(-?\d+(?:\.\d+)?)", output_string)
    return float(match.group(1)) if match else 0


def generate_eval_signal(tokenizer, model, device, news, prices, signal_strengh, threshold):
    return _generate_eval_signal(tokenizer, model, device, news, prices, signal_strengh, threshold)


def generate_signal(tokenizer, model, device, news, prices, signal_strengh, threshold):
    return _generate_signal(tokenizer, model, device, news, prices, signal_strengh, threshold)