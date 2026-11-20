# Academic Research & Survey: Benchmark Datasets for Fake Review Detection

## 1. Executive Summary

In fake review detection research, data collection is notoriously challenging because the true intention of a reviewer is rarely observable. Broadly, datasets in the literature fall into three categories:

1. **Direct Observational Ground Truth:** Datasets capturing real-world fraudulent activities (e.g., monitoring seller recruitment in underground Facebook groups or PayPal transactions).
2. **Algorithmic Filter Heuristics:** Datasets where commercial filtering systems (such as Yelp's proprietary recommendation software) designate reviews as filtered/unrecommended vs. recommended.
3. **Controlled Synthetic Crowdsourcing / Generative:** Datasets where human crowd-workers (e.g., Mechanical Turk) or language models were explicitly instructed to write deceptive reviews.
4. **Unlabeled E-Commerce Corpora:** Large-scale product review datasets that contain consumer feedback, ratings, and helpfulness votes, but **no** ground-truth deceptive/honest labels.

Below is an academic audit of seven primary candidate datasets, their licenses, schema compatibility, ground-truth provenance, and suitability for classical machine learning product review detection.

---

## 2. Comprehensive Candidate Dataset Survey

### Dataset 1: The Facebook Marketplace Amazon Fraud Dataset (Sherry He et al., 2022) — *Our Primary Dataset*
- **Primary Citation:** He, S., Hollenbeck, B., & Proserpio, G. (2022). *The Market for Fake Reviews.* Marketing Science / NBER Working Paper No. 27671.
- **Repository / Source:** [fake-reviews-data repository (Hollenbeck et al.)](https://github.com/hollenbeck/fake-reviews-data)
- **License / Access:** Academic research use (Publicly available academic release).
- **Scale:**
  - Raw: 381,734 records, 23 columns.
  - Unambiguously Labeled Cleaned Subset: **32,940 reviews** (Fake: 22,138 [67.21%], Honest: 10,802 [32.79%]).
- **Available Columns:** `asin`, `review_id`, `reviewer_id`, `review_title`, `review_text`, `review_rating`, `review_date`, `product_title`, `product_url`, `number_of_helpful`, `number_of_photos`, `fake_review_campaign_start_date`, `fake_review_product`, `reviewer_classified_fake`, `reviewer_classified_honest`, `reviewer_labeled_fake`, `reviewer_labeled_honest`, `review_is_removed_by_amazon`.
- **Label Definition & Provenance:**
  Ground-truth observational labeling. The researchers infiltrated private Facebook groups where Amazon merchants recruited individuals to buy specific ASINs and write 5-star reviews in exchange for full product refunds via PayPal plus a cash commission ($3–$10).
  - `reviewer_labeled_fake = 1`: Reviewer was directly identified inside recruitment rings posting proof of review.
  - `reviewer_labeled_honest = 1`: Control reviewers with no presence in illicit groups.
- **Suitability for Product Reviews:** **Direct match (100%).** These are real Amazon product reviews across consumer goods categories (electronics, pet supplies, home goods).
- **Limitations:** Enriched sample (higher proportion of fake reviews than natural marketplace baseline); reviewer-level labels (some individual reviews by a recruited reviewer might theoretically be candid opinions on products they liked).

---

### Dataset 2: YelpChi (Yelp Chicago)
- **Primary Citation:** Mukherjee, A., Venkataraman, V., Liu, B., & Glance, N. (2013). *Fake Review Detection: Classification and Analysis of Real and Pseudo Reviews.* UIC Technical Report / WWW 2013; Rayana, S., & Akoglu, L. (2015). *Collective Opinion Spam Detection: Bridging Review Networks and Metadata.* ACM SIGKDD 2015.
- **Source / Download:** Stony Brook University Data Mining Lab ([SBU Yelp Datasets](http://odds.cs.stonybrook.edu/yelpchi-dataset/)).
- **License:** Academic research license.
- **Scale:** 67,395 reviews across 201 hotel and restaurant businesses in Chicago written by 38,063 reviewers.
- **Class Distribution:** ~13.23% Filtered / Fake, ~86.77% Recommended / Genuine.
- **Available Columns:** `date`, `review_id`, `reviewer_id`, `prod_id`, `label` (-1 for filtered/fake, 1 for genuine), `rating`, `review_text`.
- **Label Definition & Provenance:**
  Labels are derived from Yelp's proprietary recommendation software. Yelp automatically flags reviews as "not recommended" (filtered) based on behavioral anomaly detection, IP logging, and text heuristics.
- **Suitability for Product Reviews:** **Low-to-Medium (Cross-Domain).** Covers restaurants and hotels (hospitality services), not physical e-commerce consumer goods.
- **Limitations:**
  - Yelp's filtering algorithm is proprietary and not 100% ground truth (some genuine reviews are filtered due to inactive accounts).
  - Domain shift: Restaurant reviews describe service, ambience, and taste, which have different vocabulary from product reviews (shipping, packaging, durability, specifications).

---

### Dataset 3: YelpNYC (Yelp New York City)
- **Primary Citation:** Rayana, S., & Akoglu, L. (2015). *Collective Opinion Spam Detection: Bridging Review Networks and Metadata.* ACM SIGKDD 2015.
- **Source / Download:** Stony Brook University Data Mining Lab ([SBU ODDS](http://odds.cs.stonybrook.edu/yelpnyc-dataset/)).
- **License:** Academic research use.
- **Scale:** 359,052 reviews across 923 restaurants in New York City by 160,225 reviewers.
- **Class Distribution:** ~10.27% Filtered / Fake, ~89.73% Recommended / Genuine.
- **Available Columns:** `user_id`, `prod_id`, `rating`, `label` (0/1 or -1/1), `date`, `review_content`.
- **Label Definition & Provenance:** Yelp recommendation filter (algorithmic proxy).
- **Suitability for Product Reviews:** Low-to-Medium (Service domain).
- **Limitations:** Highly sparse graph, restaurant specific, algorithmic proxy labels.

---

### Dataset 4: YelpZip (Yelp Across Continuous Zip Codes)
- **Primary Citation:** Rayana, S., & Akoglu, L. (2015). *Collective Opinion Spam Detection: Bridging Review Networks and Metadata.* ACM SIGKDD 2015.
- **Source / Download:** Stony Brook University Data Mining Lab ([SBU ODDS](http://odds.cs.stonybrook.edu/yelpzip-dataset/)).
- **License:** Academic research use.
- **Scale:** 608,598 reviews across 5,044 restaurants in contiguous zip codes across NY, NJ, PA, CT.
- **Class Distribution:** ~13.22% Filtered / Fake, ~86.78% Recommended / Genuine.
- **Available Columns:** `user_id`, `prod_id`, `rating`, `label`, `date`, `review_content`.
- **Label Definition & Provenance:** Yelp recommendation algorithm.
- **Suitability for Product Reviews:** Hospitality domain, network and metadata analysis.
- **Limitations:** Huge file size; restaurant domain shift; Yelp filtering heuristics.

---

### Dataset 5: Deceptive Opinion Spam Corpus (Ott et al. 2011, 2013)
- **Primary Citations:** 
  1. Ott, M., Choi, Y., Cardie, C., & Hancock, J. T. (2011). *Finding Deceptive Opinion Spam by Any Means Necessary.* In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).
  2. Ott, M., Cardie, C., & Hancock, J. T. (2013). *Negative Deceptive Opinion Spam.* In Proceedings of NAACL-HLT 2013.
- **Source / Download:** Cornell University NLP Group ([Ott Deceptive Corpus](https://myleott.com/op-spam/)).
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Scale:** **1,600 reviews** total (800 positive, 800 negative; 800 truthful, 800 deceptive).
- **Available Columns:** `deceptive` (`truthful` / `deceptive`), `polarity` (`positive` / `negative`), `hotel`, `review_text`.
- **Label Definition & Provenance:**
  **Gold-standard experimental ground truth.**
  - 800 truthful reviews were collected from TripAdvisor for the 20 most popular hotels in Chicago.
  - 800 deceptive reviews were generated via Amazon Mechanical Turk by crowdsourced workers who had never visited the hotels but were instructed to write realistic, convincing positive/negative reviews.
- **Suitability for Product Reviews:** **Medium (Linguistic Gold Standard).** While situated in the hotel domain, this dataset is universally cited in NLP for linguistic deception cues (e.g., pronoun rates, spatial markers, descriptive embellishments).
- **Limitations:** Small sample size (1,600 reviews); domain restricted to 20 Chicago hotels; artificially balanced 50/50 class balance.

---

### Dataset 6: Kaggle Fake Reviews Dataset (Salminen et al., 2022)
- **Primary Citation:** Salminen, J., Kandpal, C., Kamel, A. M., Jung, S. G., & Jansen, B. J. (2022). *Creating and detecting fake reviews of extreme-sentiment: an interactive and transparent approach.* Springer Nature / Electronic Commerce Research.
- **Source / Download:** Kaggle Dataset: `fake-reviews-dataset` (`https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset`).
- **License:** CC0 Public Domain.
- **Scale:** **40,432 reviews** (20,216 Original [OR] and 20,216 Computer-Generated [CG]).
- **Available Columns:** `category`, `rating`, `label` (`OR` vs `CG`), `text_`.
- **Label Definition & Provenance:**
  - `OR` (Original): Real human reviews scraped from Amazon products.
  - `CG` (Computer Generated): Synthetic reviews generated by fine-tuning GPT-2 on Amazon product review text.
- **Suitability for Product Reviews:** **High.** Directly covers Amazon physical product categories (Electronics, Home & Kitchen, Sports, Books, Clothing, etc.).
- **Limitations:** Focuses on machine-generated synthetic reviews rather than human-authored fraudulent paid reviews. Models trained on this dataset learn GPT-2 stylistic artifacts (repetition, perplexity spikes) rather than commercial incentive fraud.

---

### Dataset 7: Amazon Fine Food Reviews (`Reviews.csv`) — *Unlabeled Baseline*
- **Primary Citation:** McAuley, J., & Leskovec, J. (2013). *From amateurs to connoisseurs: modeling the evolution of user expertise through online reviews.* ACM WWW 2013.
- **Source / Download:** Stanford SNAP / Kaggle (`snap.stanford.edu/data/web-Amazon.html`).
- **License:** Open Academic / CC0.
- **Scale:** 568,454 reviews across 74,258 products written by 256,059 users.
- **Available Columns:** `Id`, `ProductId`, `UserId`, `ProfileName`, `HelpfulnessNumerator`, `HelpfulnessDenominator`, `Score`, `Time`, `Summary`, `Text`.
- **Label Definition & Provenance:**
  **CRITICAL FINDING: NO DECEPTIVE OR GENUINE LABELS EXIST.**
  This is a general customer feedback corpus. Ratings represent customer sentiment (1 to 5 stars), not authenticity.
- **Suitability for Product Reviews:** Not suitable for supervised fake review training.
- **Legitimate Academic Use:** Can be used strictly as an unlabeled out-of-domain evaluation set to demonstrate unsupervised sentiment extraction or test classifier behavior on general retail text.

---

## 3. Systematic Dataset Comparison Matrix

| Dataset Name | Domain | Scale (Rows) | Ground-Truth Basis | Fake % | Explicit Labels? | Product Suitability | Primary Academic Value |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **He et al. (Ours)** | **Amazon Products** | **32,940** | **Observed Facebook Fraud Ring** | **67.21%** | **Yes (0/1)** | **100% (Direct)** | **Real-world commercial fraud campaigns** |
| **Ott et al. (OpSpam)** | Hotels | 1,600 | Controlled Crowdsourcing (AMT) | 50.00% | Yes (0/1) | 65% (Linguistic) | Psycholinguistic gold standard |
| **Salminen et al.** | Amazon Products | 40,432 | GPT-2 Generation vs Human | 50.00% | Yes (OR/CG) | 90% (Products) | AI-generated synthetic text detection |
| **YelpChi** | Restaurants/Hotels | 67,395 | Yelp Recommendation Filter | 13.23% | Yes (-1/1) | 40% (Services) | Graph and network fraud benchmarks |
| **YelpNYC** | Restaurants | 359,052 | Yelp Recommendation Filter | 10.27% | Yes (0/1) | 35% (Services) | Large-scale urban service reviews |
| **YelpZip** | Restaurants | 608,598 | Yelp Recommendation Filter | 13.22% | Yes (0/1) | 35% (Services) | Multi-regional behavioral patterns |
| **Amazon Fine Food** | Gourmet Grocery | 568,454 | **None (Unlabeled)** | N/A | **No** | 0% (Supervised) | Unsupervised sentiment & retail shift |

---

## 4. The Unified E-Commerce Review Schema

To enable cross-dataset evaluation without corrupting label semantics or introducing artificial fields, we establish a standardized schema:

```
Unified Schema Specification:
1. review_id          : string   (Unique review identifier)
2. reviewer_id        : string   (Author / user identifier for group-splitting)
3. product_id         : string   (Product ASIN or business identifier)
4. product_category   : string   (Product category / domain; NaN if missing)
5. review_title       : string   (Review summary / title; empty string if missing)
6. review_text        : string   (Main body text of the review)
7. rating             : float    (Numeric rating, normalized to 1.0 - 5.0)
8. verified_purchase  : boolean  (True/False if confirmed; NaN if not recorded)
9. timestamp          : string   (Date or ISO timestamp; NaN if missing)
10. label             : integer  (1 = Deceptive/Fake/Filtered, 0 = Genuine/Honest)
11. original_label    : string   (Original raw label from source file)
12. source_dataset    : string   (Identifier of the originating corpus)
```

### Protocol for Handling Missing and Incompatible Data:
1. **Never Invent Labels:** If a dataset lacks ground-truth deception labels (like `Reviews.csv`), it is never mapped into the supervised target variable `label`.
2. **Preserve Raw Provenance:** Always retain `source_dataset` and `original_label` to allow auditors to trace every record back to its originating study.
3. **No Blind Concatenation:** Datasets with disparate label definitions (e.g., Yelp algorithmic filtering vs. Facebook recruitment fraud vs. GPT-2 generation) are evaluated **independently** in cross-domain experiments to measure domain shift, rather than pooled into a single contaminated training set.
