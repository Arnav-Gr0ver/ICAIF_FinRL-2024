"""
You may modify the signal generation pipeline as you wish.

We use an LLM to generate a sentiment score according to the prompt below. 

You can improve the sentiment analysis here or generate your own signal.
"""

import re
import torch

SAMPLE_PROMPT = """
Task: Evaluate the following news headline regarding a stock and assign a sentiment score between -{signal_strengh} and {signal_strengh} based on its likely impact on stock sentiment. Consider the following factors in your analysis:

- Financial Performance Indicators: Is the headline suggesting improved or weakened financials (e.g., profit, revenue, margins)? Positive indicators should increase the score, while negative indicators should lower it.

- Market Sentiment and Public Perception: Is the sentiment around the company's actions, such as innovation, partnerships, or social impact, likely to influence its stock positively or negatively?

- Competitive Positioning and Industry Impact: Does the news indicate a competitive edge or a disadvantage? For example, winning market share or regulatory issues can shift sentiment accordingly.

- Macro Events and Trends: Is the news affected by larger trends like inflation, recession fears, or industry-wide challenges? Factor these into the score where relevant.

Score interpretations:
- -{signal_strengh}: Very negative sentiment, implying substantial negative impact.
- -{threshold}: Moderately negative, potentially impacting stock with a slight downtrend.
- 0: Neutral, likely no immediate impact.
- {threshold}: Moderately positive, suggesting potential for slight growth.
- {signal_strengh}: Very positive, likely to drive significant stock appreciation.

Output only a single integer value in the range -{signal_strengh} to {signal_strengh} that best represents the sentiment.

News Headline: "{news}"
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