"""
You may modify the signal generation pipeline as you wish.

We use an LLM to generate a sentiment score according to the prompt below. 

You can improve the sentiment analysis here or generate your own signal.
"""

import re
import torch

SAMPLE_PROMPT = """Task: Analyze the following news headline about a stock and provide a sentiment score between -{signal_strengh} and {signal_strengh}. The sentiment score should predict the **short-term market movement** after the news release, considering both the **tone of the news** and the **recent market reaction**. Use the following scale:

- -{signal_strengh}: Strongly negative sentiment (likely to cause a significant price drop)
- -{threshold}: Mildly negative sentiment (could lead to a slight decline or no major change)
- 0: Neutral sentiment (no expected effect on stock price)
- {threshold}: Mildly positive sentiment (could cause a slight increase in stock price)
- {signal_strengh}: Strongly positive sentiment (likely to result in a significant price increase)

For the sentiment analysis, consider the following:
- The tone and content of the news (positive, negative, or neutral).
- How the stock has reacted to similar news in the past.
- Price movement data: Has the stock shown sensitivity to similar news in the past, or is the market generally indifferent?

Based on these factors, generate a sentiment score to predict how the stock will likely move in the next 3 days **after the news release**. The score should reflect how the news is expected to affect **short-term price fluctuations**, not just its tone.

News headline: "{news}"
Price Data: "{prices}"

Sentiment score:
"""


def _generate_signal(tokenizer, model, device, news, prices, signal_strengh, threshold):
    """Using model forward pass to do backprop"""
    prompt = SAMPLE_PROMPT.format(signal_strengh=signal_strengh, threshold=threshold, news=news, prices=prices)
    inputs = tokenizer(prompt, return_tensors="pt")  # .to(device)

    generated_ids = inputs["input_ids"]
    log_probs = []
    max_new_tokens = 5

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
    match = re.search(r"Sentiment score:\s*(-?\d+(?:\.\d+)?)", output_string)
    return float(match.group(1)) if match else 0


def generate_eval_signal(tokenizer, model, device, news, prices, signal_strengh, threshold):
    return _generate_eval_signal(tokenizer, model, device, news, prices, signal_strengh, threshold)


def generate_signal(tokenizer, model, device, news, prices, signal_strengh, threshold):
    return _generate_signal(tokenizer, model, device, news, prices, signal_strengh, threshold)