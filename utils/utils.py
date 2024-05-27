from itertools import product
import numpy as np
from typing_extensions import assert_never

from models.model_class import LanguageModel
from models.black_box_model import BlackBoxModel, CohereFamily, GPTFamily, GeminiFamily
from models.open_source_model import OpenSourceModel


def product_dict(**kwargs):
    nonempty_kwargs = {k: v for k, v in kwargs.items() if not (type(v) is list and len(v) == 0)}
    keys = nonempty_kwargs.keys()
    for instance in product(*nonempty_kwargs.values()):
        yield dict(zip(keys, instance))

def linear_probability(logprob):
    return np.round(np.exp(logprob)*100,2)

def get_max_tokens_key(model: LanguageModel):
    match model:
        case BlackBoxModel():
            return "max_tokens"
        case OpenSourceModel():
            return "max_new_tokens"
        case _ as unreachable:
            assert_never(unreachable)

def get_greedy_one_command(model: LanguageModel):
    match model:
        case GPTFamily():
            return {"n": 1, "temperature": 0}
        case GeminiFamily():
            return {"candidate_count": 1, "temperature": 0}
        case CohereFamily():
            return {"n": 1, "temperature": 0}
        case OpenSourceModel():
            return {"temperature": 0}
        case _ as unreachable:
            assert_never(unreachable)

ALL_ICL_EXEMPLARS = [
    (
        "How does artificial intelligence work?",
        "Artificial intelligence (AI) operates by analyzing large sets of data, recognizing patterns, and making decisions based on that analysis. It combines algorithms in machine learning, neural networks, and sometimes natural language processing, to mimic human cognition. AI systems learn from data inputs and improve over time, enabling them to perform tasks ranging from simple classification to complex problem-solving."
    ),
    (
        "Explain the concept of photosynthesis.",
        "Photosynthesis is the process by which green plants, algae, and certain bacteria convert light energy, usually from the sun, into chemical energy stored in glucose, a type of sugar, using water and carbon dioxide. This occurs in two main stages: the light-dependent reactions and the Calvin cycle. In the light-dependent reactions, which take place in the thylakoid membranes of chloroplasts, sunlight is absorbed by chlorophyll, causing it to release electrons that help generate ATP and NADPH. These energy carriers then power the Calvin cycle in the stroma of the chloroplast, where carbon dioxide is fixed into glucose. Oxygen is released as a byproduct of this process, contributing to the oxygen we breathe."
    ),
    (
        "Give a brief overview of the stock market.",
        "The stock market is a platform where buyers and sellers come together to trade shares of publicly listed companies. It operates through exchanges like the New York Stock Exchange or NASDAQ. Companies list their shares on an exchange through an initial public offering (IPO), allowing investors to buy and sell these shares. The price of stocks is determined by supply and demand dynamics, influenced by various factors including company performance, economic indicators, and market sentiment. Investors can make profits by buying stocks at a lower price and selling them at a higher price, or through dividends paid by the company."
    ),
    (
        "Summarize World War II.",
        "World War II was a global conflict that lasted from 1939 to 1945, involving most of the world's nations. It was primarily fought between the Axis Powers (mainly Germany, Italy, and Japan) and the Allied Powers (primarily the United Kingdom, Soviet Union, and the United States). It resulted in significant loss of life and was marked by key events like the Holocaust, the bombing of Hiroshima and Nagasaki, and the eventual defeat of Nazi Germany and Imperial Japan. World War II had profound effects on the course of world history, including the emergence of the United States and the Soviet Union as superpowers and the beginning of the Cold War."
    ),
    (
        "What are the benefits of regular physical exercise?",
        "Regular physical exercise offers a multitude of benefits for both the body and mind. Physically, it helps improve cardiovascular health, reducing the risk of heart disease, stroke, and hypertension. It aids in maintaining a healthy weight, strengthening muscles and bones, and enhancing flexibility and balance, which can prevent falls and injuries. From a mental health perspective, exercise is known to release endorphins, often referred to as feel-good hormones, which can improve mood and reduce feelings of depression and anxiety. Additionally, regular physical activity can boost cognitive function, improve sleep quality, and increase lifespan. Engaging in a variety of exercises, including aerobic activities, strength training, and flexibility exercises, can maximize health benefits and prevent the monotony of a single routine."
    ),
    (
        "Walk me through the idea of 'supply and demand' in economics.",
        "The concept of 'supply and demand' is a fundamental economic model that describes how the price and quantity of goods and services are determined in a market. Demand refers to how much (quantity) of a product or service is desired by buyers at various prices, while supply represents how much the market can offer. The price of a good or service is determined by the point at which supply and demand meet, known as the equilibrium price. When demand increases for a fixed supply, prices rise as consumers are willing to pay more to secure the item. Conversely, if supply increases for a fixed demand, prices fall as more of the product is available. This dynamic interaction between supply and demand ensures the efficient distribution of resources in a market economy, guiding the allocation of goods and services to those who value them most."
    ),
    (
        "How do cyberattacks impact businesses?",
        "Cyberattacks can have devastating impacts on businesses, ranging from financial losses to reputational damage. Financially, the direct costs include ransom payments in the case of ransomware attacks, system restoration, and data recovery expenses. There are also indirect costs, such as operational downtime, lost productivity, and legal fees associated with breaches of customer or employee data. Beyond the tangible impacts, cyberattacks can erode customer trust and loyalty, especially if sensitive personal information is compromised. This loss of trust can lead to a decline in business and may be difficult to recover from. Furthermore, businesses may face regulatory fines if found non-compliant with data protection laws. In the long term, cyberattacks can disrupt strategic plans and lead to competitive disadvantages, as resources are diverted to recovery and strengthening cybersecurity defenses instead of growth or innovation initiatives."
    ),
    (
        "Tell me the mechanism behind how black holes form.",
        "A black hole forms from the remnants of a massive star that has ended its life cycle. When such a star depletes its nuclear fuel, it can no longer sustain the nuclear fusion reactions that counterbalance gravitational forces. As a result, the core collapses under its own immense gravity, leading to a supernova explosion that ejects the star's outer layers into space. If the remaining core mass is sufficiently large—typically more than about 2.5 times the mass of the Sun—it collapses to a point of infinite density known as a singularity, surrounded by an event horizon. This event horizon marks the boundary beyond which nothing, not even light, can escape the black hole's gravitational pull, rendering the black hole invisible and detectable only by its gravitational effects on nearby matter and radiation."
    )
]