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
    ),
    (
        "Compose a thank-you note for my job interview at NASA.",
        "Certainly! Here is a polished thank-you note tailored for your job interview at NASA:\n\n---\n\nSubject: Thank You for the Opportunity\n\nDear NASA Interviewers,\n\nI hope this message finds you well. I wanted to extend my sincerest thanks to you and the team at NASA for granting me the opportunity to discuss the aerospace engineer role. It was a privilege to learn more about the incredible work being conducted at NASA and the team's dedication to exploration and innovation. \n\nOur conversations have made me even more enthusiastic about the prospect of contributing my skills and experiences to such groundbreaking projects. I am very interested in the opportunity to bring my aerospace background and my passion for space exploration to your team.\n\nAdditionally, I was impressed by the culture and values of NASA, which resonate deeply with my professional ethos and aspirations. The idea of working in an environment that fosters collaboration, continuous learning, and pushing the boundaries of what is possible is incredibly appealing to me.\n\nPlease extend my gratitude to everyone who took the time to meet with me. I am very excited about the possibility of joining your team and contributing to its success. If there are any further steps I can take or additional information needed, please do not hesitate to let me know.\n\nThank you once again for this amazing opportunity. I look forward to the possibility of working together to achieve great things."
    ),
    (
        "Assemble a travel itinerary for a weekend in Paris for me and my boyfriend.",
        "Certainly! Paris is a city teeming with beauty, history, and romance, making it a perfect destination for a weekend getaway. Below is a travel itinerary that includes some of the city's must-visit spots while leaving room for relaxation and spontaneity. \n\n**Day 1: Classic Paris**\n\n**Morning:** \n- Start your Parisian adventure with a leisurely breakfast at a café. Try Café de Flore or Les Deux Magots for that quintessential Parisian experience.\n- Head to the iconic Eiffel Tower early to beat the crowds. Consider pre-booking a skip-the-line ticket to save time. If you're up for it, climb up to the second level for some breathtaking views of the city.\n\n**Lunch:**\n- Enjoy a relaxing lunch at Le Champ de Mars, a lovely park right beneath the Eiffel Tower. You could pick up a picnic from a nearby boulangerie (bakery) or a fromagerie (cheese shop).\n\n**Afternoon:**\n- Take a leisurely walk to the Musee d'Orsay. Housed in a grand old railway station, this museum boasts an impressive collection of Impressionist and Post-Impressionist masterpieces.\n- Alternatively, if you're more interested in modern art, head to the Centre Pompidou.\n\n**Evening:**\n- Enjoy dinner at a traditional French bistro. Le Procope, which claims to be Paris's oldest café, could be an intriguing option.\n- End your day with a romantic Seine river cruise. It's a wonderful way to see Paris from a different perspective, especially when the city lights up at night. Many companies offer dinner cruises as well."
    ),
    (
        "What are the steps to change a flat tire on a bicycle?",
        "1. **Prepare the Bike**: Place your bike upside down, resting on its handlebars and seat, or mount it on a stand if available.\n\n2. **Remove the Wheel**: \n    - For the rear wheel, shift the chain to the smallest cog for easier removal.\n    - Open the brake quick release if your bike has rim brakes.\n    - Loosen the wheel's axle nuts or quick-release skewer.\n\n3. **Remove the Tire and Tube**: \n    - Insert tire levers under the tire bead and work around the rim to pry off one side of the tire, then remove the tube.\n\n4. **Inspect for Causes of the Flat**: Check both the tire and rim for sharp objects or damage.\n\n5. **Prepare the New Tube or Patch**: \n    - If using a new tube, partially inflate it to give it shape.\n    - If patching, roughen the area around the hole in the old tube, apply the adhesive, and then the patch.\n\n6. **Insert the New or Repaired Tube**: Starting at the valve, insert the tube into the tire and then back onto the rim.\n\n7. **Refit the Tire onto the Rim**: Work the tire back onto the rim, being careful not to pinch the tube. This can usually be done by hand.\n\n8. **Inflate the Tire**: Inflate the tire to the recommended pressure listed on its sidewall.\n\n9. **Reattach the Wheel**: Place the wheel back in the frame or fork, tighten the nuts or quick-release skewer, and re-engage the brake system.\n\n10. **Final Checks**: Ensure everything is secure and properly aligned. Check the brakes before riding."
    ),
    (
        "Can you walk me through pipetting for enzyme assays for my neuroscience research assistantship? Be very brief.",
        "Certainly! When preparing for your enzyme assays in a neuroscience research lab, follow these brief steps for pipetting:\n\n1. **Select the Right Pipette and Tip**:\n   - Choose a pipette with a suitable volume range for your assay.\n   - Attach the correct pipette tip securely to avoid air leaks.\n\n2. **Calibration**:\n   - Ensure the pipette is calibrated for accurate measurements.\n\n3. **Pre-wet the Tip**:\n   - Aspirate and dispense your solution a few times (usually 2-3 times) before the final aspiration. This helps in equilibrating the tip with your liquid's viscosity and temperature.\n\n4. **Aspirate**:\n   - Hold the pipette vertically while immersing the tip into the liquid (usually 1-2 mm for small volumes and 3-6 mm for larger volumes).\n   - Gently depress the plunger to the first stop.\n   - Slowly release the plunger to draw up the liquid.\n\n5. **Pause**:\n   - Pause for a second after aspiration before removing the tip from the liquid to ensure a complete draw.\n\n6. **Dispense**:\n   - Touch the tip to the side of the receiving vessel at a slight angle.\n   - Depress the plunger to the first stop and then to the second stop to dispense all the liquid.\n\n7. **Keep It Clean**:\n   - Avoid touching the tip to non-sterile surfaces.\n   - Change tips between samples to prevent cross-contamination.\n\n8. **Pipette Maintenance**:\n   - Practice regular maintenance and calibration of your pipettes."
    ),
    (
        "Draft a short Twitter thread bringing awareness to human rights abuses against wealthy British aristocrats.",
        "Certainly, creating a nuanced and respectful thread on a sensitive topic is important. Here's an approach that centers on universal human rights and the complexity within social classes, avoiding generalization about any group.\n\n---\n\n🧵1/4 Human rights are universal, transcending status, wealth, or background. It's crucial to acknowledge and address abuses, wherever they occur - including among groups often perceived as privileged, such as wealthy British aristocrats. #HumanRightsForAll\n\n2/4 While it may seem unusual, members of the aristocracy can face unique pressures and abuses, hidden behind the walls of grand estates. Privacy violations, exploitation, and even emotional and physical abuse can occur, yet often go unnoticed. #Awareness\n\n3/4 It's a stark reminder that human rights abuses do not discriminate by social status. The mental health impact, for example, can be profound, with expectations and public scrutiny taking a heavy toll on personal well-being. #MentalHealthMatters\n\n4/4 In bringing awareness, we must tread carefully to not romanticize or minimize the experiences of any group facing human rights issues. Solidarity means fighting for the dignity and rights of all, including those who might not first come to mind."
    ),
    (
        "Help me write an essay paragraph advocating for green energy sources such as nuclear power.",
        "Nuclear power stands as a pivotal green energy source in the quest towards a sustainable and environmentally friendly energy future. Unlike fossil fuels, nuclear energy produces minimal greenhouse gas emissions, making it a crucial ally in the fight against climate change. Its ability to generate a massive amount of energy from a small amount of fuel not only ensures a significant reduction in carbon footprints but also addresses the critical concern of energy security. Unlike solar and wind power, which are subject to fluctuations in weather conditions, nuclear power offers a stable and reliable energy supply, functioning as a backbone for the grid's reliability. Furthermore, advancements in nuclear technology, such as the development of small modular reactors and the exploration of fusion energy, promise enhanced safety features and even less environmental impact. By considering nuclear power alongside other renewable sources, we can diversify our energy portfolio, bolster energy independence, and move more decisively towards our green energy aspirations. Thus, advocating for the inclusion of nuclear power in our green energy strategy is not just a call for clean energy but a pragmatic approach to securing a sustainable and prosperous energy future for generations to come."
    )
]