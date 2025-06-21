import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("gr_api_key"))

prompt = """You are a fraud detection AI trained on Ethereum blockchain transaction data. You will receive a list of 18 features representing the behavior of a blockchain address. Based on this, identify if the address is involved in any type of fraudulent activity.

The input features (in order) are:

1. Avg min between sent txns
2. Avg min between received txns
3. Time difference between first and last txn (mins)
4. Sent txns count
5. Received txns count
6. Number of created contracts
7. Max value received (ETH)
8. Avg value received (ETH)
9. Avg value sent (ETH)
10. Total Ether sent (ETH)
11. Ether balance (ETH)
12. ERC20 total Ether received
13. ERC20 total Ether sent
14. ERC20 Ether sent to contracts
15. Unique ERC20 recipient addresses
16. Unique ERC20 token names received
17. Most sent token type
18. Most received token type

---

You must output a single sentence in the following format:

"Possible fraud type could be `<fraud type>`."

Examples of fraud types: Drainer,Phishing,Scam Token Distributor,Sybil Attack,Money Laundering,Bot Activity,Contract Abuse,Normal

**Input format:**
`[0.5, 25, 25.0, 156, 1, 3, 0.02, 1.22, 0.0002, 0.012, 0.00002, 20.5, 20.0, 19.3, 180, 1, "CONTRACT", "SMALL_TX"]`"""

def reason(val):
    chat_hist = [{"role": "system", "content": prompt},
        {"role": "user", "content": val}]
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=chat_hist,
            temperature=0.2,
            max_tokens=512,
        )

        res = completion.choices[0].message.content

        return res
        
    except Exception as e:
        return str(e)