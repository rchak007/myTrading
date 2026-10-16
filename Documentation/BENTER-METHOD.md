# The Bill Benter Expected-Value Method Applied to Stocks

**A practical guide based on Bill Benter’s horse-racing approach, mapped to equity analysis, with worked examples from real market data (as of mid-October 2026).**

---

> ## Before you use the numbers in §4
>
> **Stored here 2026-10-16**, as Chakravarti supplied it. The framework in §1–3
> and §5 is durable. **The prices, targets and gaps in §4 are a snapshot of
> 15–16 October 2026** and start going stale immediately — they are worked
> examples of the method, not a current screen. Re-run the process rather than
> reading those tables as live.
>
> **How this sits beside the other two framework docs:**
>
> | | asks |
> |---|---|
> | `HOW-TO-CATCH-WINNING-STOCK.md` | *is this a good business, early?* |
> | **this** | *is it cheap enough relative to what it is worth?* |
> | `WALL-STREET-LEVEL-STOCK-ANALYSIS.md` | the house rules and tiers they feed |
>
> They answer different questions and a name can pass one and fail another —
> §4.2 has AAPL as high quality with no gap, which is exactly that case.
>
> **The caveat the document makes itself, worth repeating up front:** consensus
> targets are not an edge. Everybody sees them. Benter's edge was a *better
> probability estimate than the crowd's*, and §5's "record and review" step is
> the only thing that tells you whether yours is.

---

## 1. Who Bill Benter Was and What He Actually Did

Bill Benter is a mathematician and former professional gambler. After being banned from casinos for card counting in the 1980s, he moved to Hong Kong and built one of the most successful quantitative betting operations in history. Estimates of his lifetime winnings from Hong Kong horse racing are in the hundreds of millions to nearly $1 billion.

His core insight was simple:

> **Do not try to pick winners. Try to find situations where your estimate of the probability is better than the probability implied by the market price.**

He published a 1994 paper titled *Computer Based Horse Race Handicapping and Wagering Systems: A Report* describing the framework. Key elements:

- Build a **fundamental model** (multinomial logit / probit) that estimates each horse’s true win probability from dozens (later 100+) variables (form, speed figures, jockey, track conditions, weight, etc.).
- Incorporate the **public odds** themselves as an additional input (the crowd already embeds private information).
- Only bet when the model’s probability implies **positive expected value (EV)** after the track takeout (~17–19%).
- Size bets using a disciplined method (related to the Kelly criterion).

The famous simplified formula circulating online is:

$$
EV = p \cdot b - (1 - p)
$$

Where:
- \( p \) = your estimated probability of winning
- \( b \) = net odds (profit per unit staked if you win)
- You only bet when \( EV > 0 \)

This is the foundation we mapped to stocks.

---

## 2. Mapping the Benter Method to Stocks

Horse racing and stocks are different markets, but the logic is analogous:

| Horse Racing | Stocks |
|--------------|--------|
| Public odds / tote | Current market price |
| Track takeout (~17–19%) | Transaction costs, slippage, taxes, opportunity cost |
| Fundamental model probability | Your (or a model’s) estimate of fair value / probability of scenarios |
| Positive EV bet | Buying when modeled fair value significantly exceeds price |
| Only bet overlays | Only invest when there is a clear margin of safety / edge |
| Kelly sizing | Fractional Kelly or risk-based position sizing |

### Practical Translation for Stocks

1. **Market price = the “odds”.**  
   The current price embeds the market’s collective probability-weighted view of future cash flows, growth, and risk.

2. **Build or use a probability model of value.**  
   This can be:
   - Scenario analysis (bull / base / bear cases with probabilities)
   - Discounted cash flow (DCF)
   - Analyst consensus price targets
   - Independent fair-value estimates (e.g., Morningstar)
   - Your own proprietary model

3. **Calculate the gap.**  
   If the probability-weighted expected value (or consensus fair value) is meaningfully higher than the current price after costs and risk, there is positive expected value.

4. **Demand a margin of safety.**  
   Because equities are more efficient than Hong Kong tote pools and models are imperfect, a small gap is usually not enough. Larger gaps + higher model confidence = stronger signal.

5. **Size the position for uncertainty.**  
   Even with positive EV, high uncertainty (binary outcomes, execution risk) requires smaller position sizes.

**Important caveat:** Public analyst targets and Morningstar fair values are *proxies* for a model, not a proprietary edge like Benter’s. They are useful for illustration and screening, but a real long-term edge requires better data, better models, or better calibration than the average market participant.

---

## 3. Step-by-Step Process Used in the Analyses

For every stock we examined, the process was:

1. **Get the current market price** (closing or pre-market).
2. **Collect published valuation anchors**:
   - Wall Street consensus average / median 12-month price targets
   - Number of analysts and rating distribution (Buy / Hold / Sell)
   - Independent estimates (Morningstar fair value when available)
   - Range of targets (high / low) to gauge disagreement
3. **Calculate the gap** (implied upside or downside).
4. **Assess uncertainty**:
   - Width of the target range
   - Dependence on speculative optionality (robotaxis, AI scale, etc.)
   - Business quality and cash flow predictability
   - Recent events (earnings, investor days, deliveries)
5. **Judge the EV signal**:
   - Large, consistent gap + Strong Buy consensus + relatively tighter range → stronger positive EV signal
   - Small gap or wide disagreement → weak or no edge
6. **Note limitations** and never treat the output as a recommendation.

---

## 4. Worked Examples from the Conversation

### 4.1 Marvell Technology (MRVL) — Clearest Positive EV in the Set

- **Price at analysis**: ~$241
- **Context**: Recent Investor Day (early October 2026) raised long-term revenue targets significantly (toward $70–90B by FY31 range in some reports).
- **Consensus / Fair Value**:
  - Street targets roughly $327–340
  - Morningstar ~$360
- **Implied upside**: ~35–50%
- **EV Assessment**: **Clearer positive gap** than most names examined. The Investor Day provided fresh fundamental information that models could incorporate, creating a measurable disconnect between price and updated fair-value estimates.
- **Uncertainty**: Still high (semiconductor growth execution, AI demand durability), but the gap was large enough and recent enough to stand out.
- **Conclusion in analysis**: One of the stronger positive-EV setups under the public-model lens at the time.

### 4.2 Apple (AAPL) — Little or No Positive EV

- **Price at analysis**: ~$342
- **Consensus targets**: Roughly $329–340
- **Morningstar**: Around $290 (below market)
- **Implied upside**: Flat to slightly negative vs. consensus
- **EV Assessment**: **Little to no positive expected value** on average published models. The stock was trading near or above consensus. High business quality, but the price already reflected much of the optimism.
- **Conclusion**: Not a positive-EV opportunity under the framework at the prices examined.

### 4.3 Tesla (TSLA) — Modest Gap, Very High Uncertainty

- **Price**: Closed ~$362 on October 15, 2026 (pre-market higher).
- **Market cap**: ~$1.4 trillion
- **Analyst consensus**: Average targets roughly $383–412 (~+6% to +14%). Ratings mixed (Hold to Buy). Extremely wide range (lows near $25, highs $550–600+).
- **Morningstar**: Fair value **$450**. 3-star rating (slightly undervalued). Narrow moat, very high uncertainty. A large portion of valuation driven by FSD/robotaxi (~50%) and Optimus (~30%).
- **EV Assessment**: Modest positive EV in Morningstar’s model and some Street averages. The gap is smaller and less decisive than MRVL’s. The bull case is heavily dependent on successful autonomy and robotics commercialization.
- **Conclusion**: Borderline. Positive only if your own probability on FSD/Optimus is at least as high as the models embed. High binary risk.

### 4.4 SpaceX (SPCX) — Large Street Gap, Extreme Uncertainty

- **Price**: Closed ~$149 on October 15, 2026.
- **Background**: IPO June 12, 2026 at $135. 52-week range roughly $105–$226. Market cap ~$2 trillion.
- **Analyst consensus**: Strong/Moderate Buy. Average targets ~$221–232 (~+48–55% upside). Wide range (lows ~$75–118, highs $450–800).
- **Independent views**: Some early post-IPO analyses (including Morningstar-style work) put fair value much lower (in the $60s range in certain reports).
- **EV Assessment**: Large apparent positive EV on Street models. More speculative than MRVL because the company is early post-IPO, capital-intensive, currently unprofitable on a trailing basis, and highly dependent on Starlink growth, Starship cadence, and AI/compute optionality.
- **Conclusion**: Larger Street gap than TSLA, but valuation disagreement is extreme. Edge is highly model-dependent.

### 4.5 Broader Nasdaq Comparison (October 15–16, 2026 Data)

| Stock | Approx. Price | Consensus Avg. PT | Implied Upside | Consensus Rating | EV Signal | Relative Uncertainty |
|-------|---------------|-------------------|----------------|------------------|-----------|----------------------|
| **NVDA** | ~$217 | ~$322–329 | ~48–51% | Strong Buy | Large positive gap | High (AI demand, competition) |
| **AVGO** | ~$344 | ~$528–530 | ~53% | Strong Buy | Large positive gap | High (AI custom silicon) |
| **GOOGL** | ~$349 | ~$425–434 | ~22–24% | Strong Buy | Solid positive gap | Medium-high |
| **MSFT** | ~$539 | ~$590–596 | ~10% | Strong Buy | Modest positive gap | Medium |
| **AMD** | ~$562 | ~$620–636 | ~10–13% | Strong Buy | Modest positive gap | High |
| **MRVL** | ~$241 | ~$330–360 | ~35–50% | Strong/Moderate Buy | Clearer gap | High |
| **TSLA** | ~$362 | ~$383–412 (MS $450) | Modest | Hold/Buy | Modest | Very high |
| **SPCX** | ~$149 | ~$221–232 | ~50% | Strong/Moderate Buy | Large Street gap | Very high |
| **AAPL** | ~$342 | ~$329–340 | Flat to slight negative | Moderate Buy | Little/no gap | Medium |

**Ranking by size and consistency of public-model gap (illustrative only):**

1. AVGO / NVDA (largest, most consistent Street gaps)
2. GOOGL
3. MRVL
4. MSFT / AMD
5. TSLA / SPCX (larger but more binary / contested)
6. AAPL (little or no gap)

---

## 5. How to Apply This Yourself Going Forward

### Step-by-Step Checklist

1. **Define your model or anchors**
   - Use consensus targets + independent fair values as a starting screen.
   - Better: Build your own scenario-weighted expected value (bull/base/bear with probabilities you assign).
   - Best: Develop a repeatable, testable fundamental model and track its out-of-sample accuracy.

2. **Measure the gap**
   - Calculate percentage upside to average/median target or your probability-weighted value.
   - Require a larger gap when uncertainty is high.

3. **Stress-test the assumptions**
   - What has to go right for the higher valuation to be realized?
   - How sensitive is the fair value to growth rates, margins, or discount rates?
   - Is the gap driven by recent, high-quality information (earnings, guidance, Investor Day) or by lagging estimates?

4. **Assess disagreement**
   - Wide target ranges = higher uncertainty → demand a bigger gap or smaller position.
   - Tight consensus around a large gap is more interesting than a large average driven by a few extreme bulls.

5. **Position sizing (Kelly-inspired)**
   - The simplified Kelly fraction for a binary bet is \( f^* = \frac{bp - q}{b} \), where \( q = 1 - p \).
   - In practice for stocks, use **fractional Kelly** (e.g., ¼ or ½ Kelly) or risk-based sizing (limit position so a full loss of the thesis doesn’t damage the portfolio).
   - High-uncertainty names (TSLA, SPCX, high-growth semis) should be sized smaller even if the gap looks large.

6. **Record and review**
   - Keep a log of the price, the model/target used, the gap, and the outcome over time.
   - This is how you calibrate whether your process has any edge.

### Key Limitations to Remember

- Public consensus targets are **not** a proprietary edge. Everyone sees them. The edge, if any, comes from better probability estimation or better timing of when the market underreacts to new information.
- Markets are more efficient than 1980s–1990s Hong Kong tote pools. Edges are smaller and decay faster.
- Models can be wrong for long periods. Positive EV does not guarantee short-term profits.
- Transaction costs, taxes, and opportunity cost matter.
- This is **not investment advice**. It is a framework for thinking about mispricing.

---

## 6. Summary Takeaways

- **Benter’s method** = estimate true probabilities better than the market, then only act when expected value is positive.
- **In stocks** this becomes: compare a probability-weighted fair value (or high-quality published anchors) to the current price and demand a sufficient gap given the uncertainty.
- **In the October 2026 examples**:
  - Strongest public-model gaps: **NVDA**, **AVGO**, then **GOOGL** and **MRVL**.
  - Modest or contested: **TSLA**, **SPCX**.
  - Little/no gap: **AAPL**.
- The framework is most useful as a **screening and discipline tool**, not as a mechanical buy signal. The real work is improving the quality of your probability estimates over time.

---

**Document created**: October 16, 2026  
**Data as of**: Approximately October 15–16, 2026 market closes and pre-market levels.  
**Purpose**: Educational illustration of the expected-value framework applied to public market data. Not investment advice.

---

*Use this document as a template. Update the numbers with current prices and targets whenever you run the process on a new stock or re-evaluate an old one.*
