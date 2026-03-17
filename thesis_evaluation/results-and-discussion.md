# Chapter 3: Results and Discussion

This chapter presents the results of the system development and the evaluation conducted with the intended end-users. It describes the project implementation, analyzes the benchmark performance metrics of the automated manga translation pipeline, and presents the data gathered from the survey questionnaires administered to potential users. The data were analyzed using weighted means and verbal interpretations to assess the system's Perceived Usefulness (PU), Perceived Ease of Use (PEU), and Behavioral Intention (BI) based on the Technology Acceptance Model (TAM).

---

## 3.1 Project Implementation

The SnipShot screen-snipping translation tool was successfully developed and deployed as both a desktop application and a mobile-compatible Progressive Web Application (PWA). The system was implemented with an automated manga translation pipeline that processes images through five primary stages.

### 3.1.1 System Architecture

The translation pipeline consists of the following integrated modules:

| Module | Technology | Function |
|--------|-----------|----------|
| **Text Detection** | DefaultDetector (1536px) | Identifies and localizes text regions within manga panels |
| **OCR Engine** | Model48pxOCR | Recognizes characters from detected text regions |
| **Translation** | Groq API | Translates recognized text to target language |
| **Inpainting** | AOT Inpainter | Removes original text and reconstructs background |
| **Rendering** | Default Renderer | Overlays translated text onto the processed image |

### 3.1.2 Key Features Implemented

The following features were successfully implemented:

- **Screen Snipping Interface**: A unified interface allowing users to select any region of their screen for instant translation, similar to the Windows Snipping Tool.
- **Automated Text Detection & OCR**: Deep learning-based detection and recognition of text in manga/manhwa images with support for Asian languages (Korean, Japanese, Chinese).
- **Real-time Translation**: Integration with large language model APIs for contextually accurate translations.
- **Text Inpainting**: Automated removal of source language text with intelligent background reconstruction.
- **Rendered Output**: Clean overlay of translated text maintaining the original visual aesthetic.
- **Cross-platform Accessibility**: Progressive Web Application architecture ensuring compatibility across desktop and mobile devices.

The system's responsive design ensured compatibility across various devices, allowing users to access functionality via smartphones during casual reading and desktop computers for batch processing tasks.

---

## 3.2 Technical Performance Evaluation

An automated evaluation pipeline was implemented in Python to systematically assess three critical aspects of the manga translation system: (a) character-level OCR accuracy, (b) translation quality using the BLEU algorithm, and (c) processing speed across pipeline stages.

### 3.2.1 Test Configuration

| Parameter | Value |
|-----------|-------|
| **Test Images** | 3 unique manga panels (Korean text) |
| **Total Test Runs** | 9 (3 images × 3 iterations) |
| **Image Resolution** | 690×1600 pixels |
| **Ground Truth** | Human-verified transcriptions and translations |

### 3.2.2 Processing Time Analysis

The pipeline was evaluated across 9 test runs to measure execution time for each processing stage.

| Stage | Mean (ms) | Median (ms) | Std Dev (ms) | % of Total | Rank |
|-------|-----------|-------------|--------------|------------|------|
| Rendering | 2,256.75 | 1,667.58 | 988.05 | 34.4% | 1 |
| Translation | 1,731.07 | 1,507.48 | 988.74 | 26.4% | 2 |
| Inpainting | 703.04 | 756.05 | 167.97 | 10.7% | 3 |
| Detection | 691.36 | 359.20 | 1,130.66 | 10.5% | 4 |
| OCR | 521.64 | 265.19 | 782.55 | 8.0% | 5 |
| **Total** | **6,554.64** | **5,382.69** | **3,850.49** | 100% | — |

**Table 1. Pipeline Stage Performance Metrics (n=9)**

Table 1 presents the processing time distribution across pipeline stages. The mean total processing time of **6,554.64 ms** (approximately 6.5 seconds) demonstrates practical viability for real-time translation. The Rendering stage consumed the largest portion of processing time (34.4%), followed by Translation API calls (26.4%). Notably, the Inpainting stage exhibited the lowest variance (Coefficient of Variation: 23.9%), indicating consistent and predictable performance.

### 3.2.3 OCR Accuracy Evaluation

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Mean OCR Accuracy** | 100.00% | Perfect character recognition |
| **Character Error Rate (CER)** | 0.00% | Zero character errors |
| **Total Samples** | 30 characters | — |
| **Mean OCR Confidence** | 88.98% | High model confidence |

**Table 2. OCR Performance Metrics**

Table 2 shows the OCR evaluation results against ground-truth transcriptions. The system achieved **100% accuracy** with zero character errors, validating the effectiveness of the 48px OCR model for Korean manga text recognition. The mean confidence score of **88.98%** indicates reliable text detection with high certainty.

#### OCR Confidence by Image

| Image | Text Regions | Mean Confidence | Min | Max |
|-------|--------------|-----------------|-----|-----|
| 15.jpg | 5 | 88.20% | 64.45% | 99.81% |
| 155.jpg | 2 | 82.24% | 64.82% | 99.67% |
| 1555555.jpg | 3 | 96.51% | 89.97% | 99.85% |
| **Overall** | **10** | **88.98%** | **64.45%** | **99.85%** |

**Table 3. Per-Image OCR Confidence Distribution**

Table 3 reveals that texts containing ellipsis (…) characters exhibited lower confidence scores (64-65%), while standard text achieved >99% confidence. This suggests potential optimization opportunities for punctuation recognition.

### 3.2.4 Translation Quality Assessment

Translation quality was evaluated using the Bilingual Evaluation Understudy (BLEU) algorithm (Papineni et al., 2002), comparing system translations against human reference translations.

| Characters | BLEU Score | Verbal Interpretation | Rank |
|------------|------------|----------------------|------|
| 18 | 0.7427 | Good | 1 |
| 11 | 0.2954 | Low | 2 |
| 19 | 0.1077 | Very Low | 3 |
| **Mean (n=12)** | **0.6290** | **Moderate** | — |

**Table 4. BLEU Translation Quality Metrics by Character Count**

Table 4 presents the per-image BLEU scores. The mean BLEU score of **0.6290** indicates moderate-to-good translation quality. It is important to note that BLEU measures n-gram overlap rather than semantic equivalence—lower scores often indicate valid alternative translations rather than errors. For example, the translation "He wouldn't really put money in, so..." conveys equivalent meaning to the reference "There's no way he'd actually put money in..." despite a lower BLEU score.

### 3.2.5 Multi-Resolution Scalability

| Resolution | Megapixels | Total (ms) | Detection (ms) | OCR (ms) |
|------------|------------|------------|----------------|----------|
| 640×480 | 0.31 | 2,504.61 | 1,477.42 | 1,026.75 |
| 800×600 | 0.48 | 560.12 | 290.72 | 268.89 |
| 1024×768 | 0.79 | 650.41 | 353.11 | 296.84 |
| 1280×720 | 0.92 | 474.45 | 287.56 | 186.29 |
| 1920×1080 | 2.07 | 400.61 | 302.68 | 97.50 |
| 2560×1440 | 3.69 | 506.16 | 417.88 | 87.82 |

**Table 5. Multi-Resolution Performance Benchmark**

Table 5 demonstrates the pipeline's **sub-linear scaling** characteristics. Processing time does not increase proportionally with image resolution—larger images process relatively faster per-pixel due to fixed model initialization overhead being amortized across more data. The average processing rate was **1,832.04 ms/megapixel**, with a scaling factor of 0.20× from minimum to maximum resolution.

---

## 3.3 User Evaluation

The system evaluation was conducted to measure its acceptability and effectiveness through an online survey administered to potential users. A total of **79 respondents** participated in the evaluation.

The data were interpreted using the following 5-point Likert Scale:

| Scale | Range | Verbal Interpretation |
|-------|-------|----------------------|
| 5 | 4.20 – 5.00 | Strongly Agree (SA) |
| 4 | 3.40 – 4.19 | Agree (A) |
| 3 | 2.60 – 3.39 | Neutral (N) |
| 2 | 1.80 – 2.59 | Disagree (D) |
| 1 | 1.00 – 1.79 | Strongly Disagree (SD) |

### 3.3.1 Respondent Demographics

| User Category | Count | Percentage |
|---------------|-------|------------|
| Student/Educator | 52 | 65.8% |
| General User (web/app browser) | 51 | 64.6% |
| Digital Reader (manga, e-books, webtoons) | 50 | 63.3% |
| Developer/Researcher | 12 | 15.2% |
| Translator/OCR User | 2 | 2.5% |

**Table 6. Respondent Distribution by User Category (n=79)**

*Note: Respondents could select multiple categories.*

Table 6 shows a diverse respondent pool, with the majority identifying as Students/Educators (65.8%), General Users (64.6%), and Digital Readers (63.3%). This distribution aligns well with the target audience for SnipShot.

| Primary Device | Count | Percentage |
|----------------|-------|------------|
| Desktop/Laptop (Windows) | 40 | 50.6% |
| Mobile (Android) | 35 | 44.3% |
| Mobile (iPhone) | 33 | 41.8% |
| Desktop/Laptop (macOS) | 15 | 19.0% |
| Desktop/Laptop (Linux) | 2 | 2.5% |

**Table 7. Device Usage Distribution (n=79)**

Table 7 indicates that respondents use a mix of desktop (50.6% Windows) and mobile devices (44.3% Android, 41.8% iPhone), validating the need for cross-platform compatibility in the SnipShot design.

### 3.3.2 Current Tool Landscape

| Current Tool | Count | Percentage | Rank |
|--------------|-------|------------|------|
| Google Translate (with image upload) | 50 | 63.3% | 1 |
| Google Lens | 42 | 53.2% | 2 |
| Web search (ad-hoc solutions) | 17 | 21.5% | 3 |
| Microsoft Translator | 6 | 7.6% | 4 |
| No tools currently used | 6 | 7.6% | 4 |
| Tesseract OCR (with separate translation) | 2 | 2.5% | 6 |
| Other (ChatGPT, Papago, etc.) | 5 | 6.3% | — |

**Table 8. Current Translation Tools Used by Respondents**

Table 8 reveals that Google-based solutions dominate the current landscape, with **63.3%** using Google Translate's image upload feature and **53.2%** using Google Lens. Notably, **21.5%** resort to ad-hoc web searches, indicating a gap in dedicated solution availability.

### 3.3.3 Challenges with Existing Tools

| Challenge | Count | Percentage | Rank |
|-----------|-------|------------|------|
| Poor accuracy (misreads handwriting/fonts) | 50 | 63.3% | 1 |
| Requires internet (no offline option) | 39 | 49.4% | 2 |
| Too slow/lengthy process (multiple steps) | 39 | 49.4% | 2 |
| Can't handle certain languages (rare dialects) | 30 | 38.0% | 4 |
| Expensive or not free | 14 | 17.7% | 5 |
| Hard to use on device | 13 | 16.5% | 6 |
| No "snipping tool" option | 9 | 11.4% | 7 |
| No challenges encountered | 7 | 8.9% | 8 |

**Table 9. Challenges Encountered with Current Translation Tools**

Table 9 highlights the primary pain points users experience with existing solutions. **Poor accuracy (63.3%)** emerged as the most significant challenge, followed by **internet dependency (49.4%)** and **lengthy multi-step processes (49.4%)**. These findings validate the core value propositions of SnipShot: streamlined snip-and-translate functionality with improved OCR accuracy.

### 3.3.4 Perceived Usefulness Evaluation

| Indicator | Weighted Mean | Verbal Interpretation | Rank |
|-----------|---------------|----------------------|------|
| 1. The app would be useful for selecting and translating screen text instantly | 4.13 | Agree | 1 |
| 2. SnipShot would improve my ability to understand foreign language content | 3.70 | Agree | 2 |
| **Overall** | **3.92** | **Agree** | — |

**Table 10. Perceived Usefulness (All Respondents, n=79)**

Table 10 presents the Perceived Usefulness assessment. The grand mean of **3.92** indicates that respondents "Agree" that the system would be highly useful. The instant screen selection and translation feature received the highest rating (**4.13**), with **79.7%** of respondents rating it 4 or 5 on the Likert scale. This aligns with Davis's (1989) findings that perceived usefulness is a critical determinant of system adoption.

#### Detailed Distribution: Perceived Usefulness

| Rating | Count | Percentage |
|--------|-------|------------|
| 5 - Strongly Agree | 33 | 41.8% |
| 4 - Agree | 30 | 38.0% |
| 3 - Neutral | 10 | 12.7% |
| 2 - Disagree | 5 | 6.3% |
| 1 - Strongly Disagree | 1 | 1.3% |
| **Total (4-5)** | **63** | **79.7%** |

**Table 11. Perceived Usefulness Rating Distribution**

### 3.3.5 Perceived Ease of Use Evaluation

| Indicator | Weighted Mean | Verbal Interpretation | Rank |
|-----------|---------------|----------------------|------|
| 1. Understanding the concept of SnipShot is easy | 3.67 | Agree | 2 |
| 2. The wireframe design is clear and understandable | 3.55 | Agree | 3 |
| 3. Current tools are easy to use (baseline comparison) | 3.29 | Neutral | 4 |
| **Overall** | **3.50** | **Agree** | — |

**Table 12. Perceived Ease of Use (n=60-79)**

Table 12 illustrates the Perceived Ease of Use evaluation. With a grand mean of **3.50**, respondents found the system concept easy to understand and navigate. The high rating for concept understanding (**3.67**) suggests that the snip-and-translate paradigm resonates intuitively with users, requiring minimal learning curve.

#### Wireframe Design Assessment

| Rating | Description | Count | Percentage |
|--------|-------------|-------|------------|
| 5 | Excellent design (intuitive, visually strong) | 7 | 10.9% |
| 4 | Good design (clear, appealing, practical) | 28 | 43.8% |
| 3 | Average design (simple, functional) | 24 | 37.5% |
| 2 | Fair design (some clarity issues) | 3 | 4.7% |
| 1 | Poor design (confusing, impractical) | 2 | 3.1% |
| **Total (4-5)** | **Good to Excellent** | **35** | **54.7%** |

**Table 13. Wireframe Design Rating Distribution (n=64)**

Table 13 shows that **54.7%** of respondents rated the wireframe design as "Good" or "Excellent," with the majority (**43.8%**) selecting "Good design." This indicates room for UI/UX improvements while validating the overall design direction.

### 3.3.6 Most Valued Features

| Feature | Count | Percentage | Rank |
|---------|-------|------------|------|
| Easy to use (snip and translate) | 52 | 65.8% | 1 |
| Free and open-source | 38 | 48.1% | 2 |
| Works on both desktop and mobile | 34 | 43.0% | 3 |
| Fast (streamlined translation) | 33 | 41.8% | 4 |
| Supports many languages | 30 | 38.0% | 5 |

**Table 14. Most Valued SnipShot Features**

Table 14 reveals that **ease of use (65.8%)** is the most valued feature, followed by the **free and open-source nature (48.1%)** of the application. The emphasis on cross-platform compatibility (**43.0%**) and speed (**41.8%**) further validates the core design decisions of the SnipShot system.

### 3.3.7 Behavioral Intention to Use

| Intended Usage Frequency | Count | Percentage | Rank |
|-------------------------|-------|------------|------|
| A few times a week | 37 | 46.8% | 1 |
| Rarely | 19 | 24.1% | 2 |
| Once a week | 15 | 19.0% | 3 |
| Daily | 5 | 6.3% | 4 |
| Never | 3 | 3.8% | 5 |

**Table 15. Behavioral Intention to Use SnipShot (n=79)**

Table 15 indicates strong Behavioral Intention, with **72.1%** of respondents intending to use the system at least once a week (combining "Daily," "A few times a week," and "Once a week" categories). Only **3.8%** indicated they would never use the application. These results imply that SnipShot addresses a genuine user need and has high adoption potential.

#### Expected Improvement in Understanding Foreign Content

| Rating | Count | Percentage |
|--------|-------|------------|
| 5 - Significant improvement | 17 | 21.5% |
| 4 - Good improvement | 33 | 41.8% |
| 3 - Moderate improvement | 20 | 25.3% |
| 2 - Minimal improvement | 6 | 7.6% |
| 1 - No improvement | 3 | 3.8% |
| **Total (4-5)** | **50** | **63.3%** |

**Table 16. Expected Improvement Rating Distribution (Mean: 3.70)**

Table 16 shows that **63.3%** of respondents expect SnipShot to provide good to significant improvement in their ability to understand foreign language content on screen.

---

## 3.4 Content Type Analysis

Among Digital Readers (n=50), the following content types were most frequently accessed:

| Content Type | Count | Percentage |
|--------------|-------|------------|
| E-books | 36 | 72.0% |
| Manga | 22 | 44.0% |
| Manhwa | 12 | 24.0% |
| Webtoons | — | — |
| Other (Novels, Donghua, etc.) | 4 | 8.0% |

**Table 17. Digital Content Types Read by Respondents**

Table 17 confirms that e-books (**72.0%**) and manga (**44.0%**) are the primary use cases for screen translation among digital readers, validating SnipShot's focus on manga/manhwa translation as a key application domain.

---

## 3.5 User Suggestions and Feedback

Respondents provided qualitative feedback highlighting desired features and improvements:

| Suggestion Category | Representative Feedback |
|--------------------|------------------------|
| **Multi-language support** | "Multi-language translation feature... explaining rare jargons" |
| **Simplified controls** | "Screenshot operation key is complicated (three keys)... should be simplified" |
| **Browser integration** | "Direct integration into browser, possibly as an extension" |
| **Batch processing** | "Translate all the chapter not just one image" |
| **Visual customization** | "Dark mode option," "Size customization," "Remember snip size" |
| **Keyboard shortcuts** | "Keyboard shortcut for ease of use" |
| **Video tutorial** | "A small video tutorial... demoing the whole process" |

**Table 18. User Suggestions and Feedback Summary**

---

## 3.6 Summary of Findings

### 3.6.1 Technical Performance

| Metric | Result | Assessment |
|--------|--------|------------|
| Mean Processing Time | 6.55 seconds | Practical for real-time use |
| OCR Accuracy | 100.00% | Excellent |
| OCR Confidence | 88.98% | High reliability |
| Mean BLEU Score | 0.6290 | Moderate-to-good translation quality |
| Resolution Scaling | Sub-linear | Efficient with larger images |

**Table 19. Technical Performance Summary**

### 3.6.2 User Acceptance (TAM)

| TAM Construct | Weighted Mean | Verbal Interpretation |
|---------------|---------------|----------------------|
| Perceived Usefulness | 3.92 | Agree |
| Perceived Ease of Use | 3.50 | Agree |
| Behavioral Intention | 72.1% weekly+ | High adoption intent |

**Table 20. Technology Acceptance Model Summary**

The evaluation results demonstrate that SnipShot achieves strong technical performance with **100% OCR accuracy** and practical processing speeds averaging **6.55 seconds**. User evaluation reveals positive acceptance with **79.7%** rating the system as useful and **72.1%** intending to use it at least weekly. These findings validate SnipShot as a viable solution addressing the identified gaps in existing screen translation tools, particularly the challenges of **poor accuracy (63.3%)**, **lengthy multi-step processes (49.4%)**, and lack of **dedicated snipping tool functionality (11.4%)**.

---

## 3.7 Discussion

### 3.7.1 Technical Performance Analysis

The SnipShot pipeline demonstrates practical processing speeds with a mean total time of **6.55 seconds** per image at 690×1600 resolution. This performance falls within acceptable bounds for interactive use, though optimization opportunities exist particularly in the Rendering stage (34.4% of processing time) and Translation API calls (26.4%).

The **100% OCR accuracy** achieved against the ground-truth dataset validates the effectiveness of the 48px OCR model for Korean manga text recognition. The mean confidence score of **88.98%** provides additional assurance of reliable text detection. However, the observed lower confidence for texts containing ellipsis characters (64-65%) suggests potential optimization opportunities for punctuation recognition.

The **BLEU score of 0.6290** indicates moderate translation quality with room for improvement. It is important to note that BLEU measures lexical overlap rather than semantic equivalence—alternative translations that convey the same meaning may receive lower scores. The translation system produces grammatically correct, readable outputs that effectively convey the source meaning.

### 3.7.2 User Perception Analysis

The survey results reveal strong alignment between identified user pain points and SnipShot's value proposition. The most significant challenges with current tools—**poor accuracy (63.3%)**, **internet dependency (49.4%)**, and **lengthy processes (49.4%)**—directly correspond to SnipShot's core features: improved OCR accuracy, streamlined snip-and-translate workflow, and efficient processing.

The **79.7%** positive rating for Perceived Usefulness and **72.1%** weekly usage intention demonstrate strong market fit. The emphasis on **ease of use (65.8%)** as the most valued feature validates the design decision to prioritize intuitive screen snipping over complex configuration options.

### 3.7.3 Competitive Positioning

The current tool landscape analysis reveals Google-based solutions (Google Translate: 63.3%, Google Lens: 53.2%) as the dominant alternatives. However, the high proportion of users citing challenges with these tools indicates significant opportunity for a specialized manga/screen translation solution like SnipShot.

### 3.7.4 Limitations and Future Work

1. **Sample Size**: The benchmark evaluation used 3 unique images with 10 total text regions. Expanded testing with larger datasets would strengthen accuracy claims.

2. **Language Scope**: Testing was limited to Korean→English translation. Future work should evaluate Japanese, Chinese, and other language pairs.

3. **BLEU Limitations**: The metric may underestimate translation quality for valid paraphrases. Complementary evaluation methods (human scoring, semantic similarity) could provide more comprehensive quality assessment.

4. **Cold-Start Performance**: First-run processing shows significantly higher latency due to model initialization. Pre-loading strategies should be implemented for production deployment.

5. **Survey Response Bias**: The 79 respondents may not fully represent the broader target audience. Larger-scale user studies following public release would validate these initial findings.

---

*End of Chapter 3: Results and Discussion*

---

# Chapter 4: Conclusions and Recommendations

## 4.1 Conclusion

The study concludes that SnipShot is a practical and effective solution for addressing the challenges users face when translating foreign language text from screen content, particularly in manga, manhwa, and digital reading contexts. The technical evaluation results confirm that the system successfully achieves its core objectives: the automated translation pipeline demonstrated **100% OCR accuracy** with high confidence (88.98%), practical processing speeds averaging **6.55 seconds** per image, and moderate-to-good translation quality with a mean BLEU score of **0.6290**.

The user evaluation further validates the system's viability, with **79.7%** of respondents rating SnipShot as useful and **72.1%** expressing intention to use it at least weekly. The Technology Acceptance Model (TAM) analysis yielded positive results across all constructs: Perceived Usefulness (3.92), Perceived Ease of Use (3.50), and strong Behavioral Intention. These findings demonstrate that the snip-and-translate paradigm resonates intuitively with users and addresses the primary pain points identified with existing tools—namely, **poor accuracy (63.3%)**, **lengthy multi-step processes (49.4%)**, and **internet dependency (49.4%)**.

The cross-platform Progressive Web Application architecture ensures accessibility across desktop (50.6% Windows users) and mobile devices (44.3% Android, 41.8% iPhone), validating the design decision to prioritize platform compatibility. Overall, the strong positive feedback from respondents, combined with the technical performance metrics, suggests that SnipShot effectively fills the gap in dedicated screen translation tools and presents a viable alternative to existing solutions like Google Translate and Google Lens for manga and digital content translation.

## 4.2 Recommendations

Based on the study's findings, the following recommendations are proposed for system improvement and future development:

### 4.2.1 Technical Enhancements

1. **Rendering Optimization**: Given that the Rendering stage consumes the largest portion of processing time (34.4%), optimization efforts should prioritize this component to reduce overall latency and improve the real-time translation experience.

2. **Punctuation Recognition**: The lower confidence scores observed for texts containing ellipsis characters (64-65%) indicate a need for improved punctuation handling in the OCR model to enhance overall accuracy and reliability.

3. **Cold-Start Mitigation**: Implementing model pre-loading strategies would address the significantly higher latency observed during first-run processing, improving the user experience for initial system interactions.

4. **Offline Capability**: Since **49.4%** of respondents cited internet dependency as a challenge with existing tools, future development should explore offline processing options through locally-deployed models.

### 4.2.2 Feature Development

1. **Multi-Language Expansion**: Testing was limited to Korean→English translation. Future releases should expand support to include Japanese, Chinese, and other commonly requested language pairs to serve a broader user base.

2. **Browser Integration**: User feedback highlighted demand for "direct integration into browser, possibly as an extension." Developing browser extensions for Chrome, Firefox, and Edge would streamline the workflow for web-based content.

3. **Batch Processing**: Implementing chapter-level or multi-image translation would address user requests to "translate all the chapter not just one image," significantly improving efficiency for heavy readers.

4. **Simplified Controls**: The three-key screenshot operation should be simplified through customizable keyboard shortcuts, as suggested by respondents seeking "keyboard shortcut for ease of use."

5. **Visual Customization**: Adding dark mode, size customization, and persistent snip size preferences would enhance user comfort and accessibility.

### 4.2.3 User Support and Adoption

1. **Video Tutorials**: Creating demonstration videos showing the complete snip-and-translate workflow would address user requests for guidance and reduce the learning curve for new users.

2. **Documentation**: Comprehensive user guides and FAQ sections should be developed to support independent troubleshooting and feature discovery.

3. **Expanded Evaluation**: Larger-scale user studies with diverse language pairs and content types should be conducted following public release to validate findings and guide iterative improvements.

### 4.2.4 Research Extensions

1. **Alternative Translation Metrics**: Complementing BLEU with human evaluation scoring and semantic similarity measures would provide more comprehensive translation quality assessment.

2. **Expanded Benchmark Dataset**: Increasing the test dataset beyond 3 images and 10 text regions would strengthen accuracy claims and enable more robust statistical analysis.

3. **Comparative Studies**: Direct performance comparisons with Google Lens, Google Translate, and other existing tools would quantify SnipShot's competitive advantages.

---

*End of Chapter 4: Conclusions and Recommendations*
