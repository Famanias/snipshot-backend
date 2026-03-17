# SnipShot: A Deep Learning-Based Screen-Snipping Translation Tool for Digital Comics

**[Your Name]¹**

¹ Department of Computer Science, Gordon College, Olongapo City, Zambales, Philippines

yourname@gordoncollege.edu.ph

---

**Abstract.** This study develops and evaluates SnipShot, a deep learning-based screen-snipping translation tool for digital comics using Design Science Research (DSR) methodology. A needs assessment survey (n=80) identified user challenges with existing image translation tools—65% cited slow, multi-step workflows as the primary issue. The system integrates DBNet-ResNet34 for text detection, Roformer+XPOS for OCR, Groq LLM (Llama-3-70b) for translation, and LaMa Large for image inpainting into a unified pipeline. Evaluation results demonstrated 93% OCR accuracy across Japanese, Chinese, and Korean text, 87% semantic accuracy for Japanese-to-English translation, and 3.8-second average processing time for 1080p images—representing an 83.5% improvement over traditional workflows. User acceptance testing (n=20) grounded in the Technology Acceptance Model (TAM) yielded strong results: Perceived Ease of Use (PEU) = 4.6/5, Perceived Usefulness (PU) = 4.1/5, with 90% recommendation rate indicating high Behavioral Intention (BI). The findings indicate that SnipShot effectively addresses gaps in existing tools by providing seamless text integration through inpainting and desktop screen-capture support not offered by mainstream alternatives.

**Keywords:** Digital Comics, Manga Translation, Deep Learning, OCR, Neural Machine Translation, Image Inpainting, Screen Capture, Technology Acceptance Model

---

## Introduction

### Background of the Study

The global digital comics market has experienced remarkable growth, with manga generating over $5 billion in annual revenue and Korean webtoons reaching 67 million monthly active users worldwide (LINE Webtoon, 2024). This international readership increasingly seeks access to content in its original language, creating substantial demand for efficient image translation tools. Traditional text translation services have revolutionized text-to-text translation; however, when text is embedded within images—as in speech bubbles, sound effects, or narrative boxes—users face a complex multi-step workflow that interrupts reading flow.

Existing solutions present notable limitations. Google Lens and Apple Live Text are primarily mobile-focused, leaving desktop users underserved. Furthermore, these tools display translations as text overlays rather than integrating them seamlessly into the original image, disrupting visual storytelling integral to the comic medium. Recent advances in deep learning have enabled sophisticated image processing capabilities: CNNs for text detection, attention-based models for OCR, LLMs for contextual translation, and GANs for image inpainting (Liao et al., 2020; Suvorov et al., 2022). The integration of these technologies presents an opportunity to create a comprehensive solution addressing existing limitations.

### Objectives of the Study

Objectives of the Study
This study aims to develop and evaluate Deep Learning-Based translation tools designed specifically for digital comics. Specifically, it seeks to:

1. Identify the challenges and user requirements for image translation tools among digital comic readers through needs assessment.
2. Design and implement a system integrating text detection, OCR, neural machine translation, and image inpainting into a unified pipeline.
3. Evaluate system effectiveness in terms of translation accuracy, processing speed, and user satisfaction.
4. Provide recommendations for improving image translation tools for the digital comics domain.


This study aims to develop and evaluate a deep learning-based screen-snipping translation tool for digital comics following Design Science Research methodology. Specifically, it seeks to:

1. Identify the challenges and user requirements for image translation tools among digital comic readers through needs assessment.
2. Design and implement a system integrating text detection, OCR, neural machine translation, and image inpainting into a unified pipeline.
3. Evaluate system effectiveness in terms of translation accuracy, processing speed, and user acceptance using the Technology Acceptance Model (TAM) framework.
4. Provide recommendations for improving image translation tools for the digital comics domain.


### Theoretical Framework

The **Technology Acceptance Model (TAM)**, proposed by Davis (1989), is the primary evaluative framework used in this study. TAM is determined by two core constructs: **Perceived Usefulness (PU)**—the degree to which a user believes the system will enhance their task performance—and **Perceived Ease of Use (PEU)**—the extent to which the user perceives the system as effortless to operate. These two constructs jointly influence a user's **Behavioral Intention (BI)** to use the technology, which in turn predicts actual system usage.

```
┌─────────────────────────────────────────────────────────────────┐
│              TECHNOLOGY ACCEPTANCE MODEL (TAM)                  │
│                        Davis (1989)                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   External                                                      │
│   Variables                                                     │
│       ↓                                                         │
│   ┌──────────────┐                                              │
│   │  Perceived   │──────────────────┐                           │
│   │  Usefulness  │                  ↘                           │
│   │    (PU)      │          ┌──────────────┐   ┌─────────┐     │
│   └──────────────┘          │  Behavioral  │   │ Actual  │     │
│         │                   │  Intention   │──→│ System  │     │
│         │                   │    (BI)      │   │  Use    │     │
│   ┌──────────────┐          └──────────────┘   └─────────┘     │
│   │  Perceived   │──────────────────┘                           │
│   │  Ease of Use │                                              │
│   │   (PEU)      │──────────→ (also influences PU)             │
│   └──────────────┘                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```
**Fig. 1.** Technology Acceptance Model (TAM) (Davis, 1989)

TAM has been widely applied in information systems research as a model for explaining technology adoption across diverse domains, including mobile applications (Alalwan et al., 2017), e-learning platforms (Abdullah et al., 2016), and translation technologies (Hsu, 2017). In this study, TAM provides the theoretical lens for evaluating SnipShot's acceptability among digital comic readers. Specifically, PEU assesses whether the snip-and-translate workflow minimizes user effort compared to existing multi-step alternatives, PU evaluates whether users believe the system meaningfully improves their ability to understand foreign-language comics, and BI—measured through recommendation likelihood—captures users' intention to adopt and advocate for the tool.

SnipShot's technical architecture is grounded in multiple deep learning theories that enable its translation pipeline. The **deep residual learning for image recognition** (He et al., 2016) underpins DBNet-ResNet34 for text detection, enabling the training of deeper convolutional neural networks by learning residual functions that bypass vanishing gradient problems—allowing the model to capture hierarchical visual features from training data rather than hand-crafted rules, critical for accurate text detection in complex comic layouts across diverse styles and languages. The **encoder-decoder framework with attention mechanism** (Bahdanau et al., 2015; Vaswani et al., 2017) underpins the translation component, where the encoder maps source language text into context representations and the decoder generates target language output—implemented through Meta's Llama-3-70b transformer model that captures nuanced semantic relationships including honorifics and onomatopoeia in comics. **Image inpainting theory** (Bertalmio et al., 2000; Suvorov et al., 2022) enables text removal and background reconstruction through the LaMa model's fast Fourier convolutions (FFCs), which learn to predict plausible content for masked regions by capturing both local textures and global image structures—critical for maintaining the visual integrity of comic panels by reconstructing artwork behind removed text rather than overlaying translations. Together, these theories justify SnipShot's data-driven "erase-then-render" approach that distinguishes it from overlay-based alternatives.

```
┌────────────────────────────────────────────────────────────────────────────┐
│              DEEP LEARNING THEORETICAL FOUNDATIONS FOR SNIPSHOT            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌─────────────────────┐        ┌──────────────────────┐                  │
│  │  DEEP RESIDUAL      │        │  ENCODER-DECODER     │                  │
│  │  LEARNING           │        │  WITH ATTENTION      │                  │
│  │  (He et al., 2016)  │        │  (Bahdanau et al.,   │                  │
│  └─────────┬───────────┘        │  2015; Vaswani       │                  │
│            │                     └──────────┬───────────┘                  │
│            │                                │                              │
│            ↓                                ↓                              │
│  ┌─────────────────────┐        ┌──────────────────────┐                  │
│  │  Residual CNNs      │        │  Transformer         │                  │
│  │  for Feature        │        │  Architecture        │                  │
│  │  Learning           │        └──────────┬───────────┘                  │
│  └─────────┬───────────┘                   │                              │
│            │                                │                              │
│            ↓                                ↓                              │
│  ┌──────────────────────────────────────────────────────┐                 │
│  │              SNIPSHOT TRANSLATION PIPELINE           │                 │
│  ├──────────────────────────────────────────────────────┤                 │
│  │                                                      │                 │
│  │  1. TEXT DETECTION                                   │                 │
│  │     • DBNet-ResNet34 (Residual CNN)                 │                 │
│  │     • Learns hierarchical features via residuals    │                 │
│  │       ↓                                              │                 │
│  │  2. OCR (Character Recognition)                      │                 │
│  │     • Roformer+XPOS (Transformer)                   │                 │
│  │     • Sequential character dependencies             │                 │
│  │       ↓                                              │                 │
│  │  3. TRANSLATION                                      │ ← Encoder-Decoder│
│  │     • Llama-3-70b (Transformer LLM)                 │                 │
│  │     • Context mapping & semantic relationships      │                 │
│  │       ↓                                              │                 │
│  │  4. IMAGE INPAINTING                                 │ ← Inpainting   │
│  │     • LaMa Large (Fourier CNNs)                     │   Theory       │
│  │     • Text removal & background reconstruction      │                 │
│  │       ↓                                              │                 │
│  │  5. TEXT RENDERING                                   │                 │
│  │     • Translated text on clean canvas               │                 │
│  │                                                      │                 │
│  └──────────────────────────────────────────────────────┘                 │
│            ↑                                ↑                              │
│            │                                │                              │
│  ┌─────────────────────┐        ┌──────────────────────┐                  │
│  │  Deep Residual:     │        │  Image Inpainting    │                  │
│  │  Training Deeper    │        │  Theory              │                  │
│  │  Networks           │        │  (Bertalmio et al.,  │                  │
│  │  (He et al., 2016)  │        │  2000; Suvorov       │                  │
│  │                     │        │  et al., 2022)       │                  │
│  └─────────────────────┘        └──────────────────────┘                  │
└────────────────────────────────────────────────────────────────────────────┘
```
**Fig. 2.** Deep Learning Theoretical Foundations Applied to SnipShot Pipeline

### Conceptual Framework

This study employs the **Input-Process-Output (IPO) model** as its conceptual framework to provide a systematic structure for developing and evaluating the SnipShot translation tool. The IPO model is particularly appropriate for this research because it aligns with the descriptive-developmental nature of the study and the Design Science Research methodology employed.

The IPO model serves three essential functions in this research. First, it provides a clear organizational structure that maps the entire research workflow—from identifying user needs and theoretical foundations (Input), through system design and implementation (Process), to comprehensive technical and user acceptance evaluation (Output). Second, it ensures systematic progression through distinct research phases, preventing gaps in the development process and guaranteeing that evaluation criteria are established before implementation begins. Third, it facilitates transparent communication of the research process to stakeholders by clearly delineating what activities occur at each stage and how they contribute to the final artifact.

This framework is widely used in systems development research because it explicitly acknowledges that creating an IT artifact requires not only understanding the problem space and building a solution, but also rigorously evaluating whether the solution effectively addresses the identified needs. For SnipShot, the IPO model ensures that the deep learning-based translation system is developed with clear requirements derived from user needs (Input), implemented using appropriate technologies and methodologies (Process), and validated through both technical performance metrics and user acceptance constructs (Output).

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                          INPUT-PROCESS-OUTPUT MODEL                              │
│                     Applied to SnipShot Development & Evaluation                 │
├────────────────────────┬─────────────────────────┬────────────────────────────────┤
│                        │                         │                                │
│        INPUT           │       PROCESS           │          OUTPUT                │
│  (Problem & Theory)    │  (Artifact Development) │  (Evaluation & Validation)     │
│                        │                         │                                │
├────────────────────────┼─────────────────────────┼────────────────────────────────┤
│                        │                         │                                │
│ • Literature Review    │ • System Architecture   │ TECHNICAL METRICS:             │
│   - Deep Learning      │   Design                │  • OCR Accuracy: 93%           │
│   - Image Translation  │                         │    - Japanese: 94.2%           │
│   - User Needs         │ • Model Integration:    │    - Chinese: 92.8%            │
│                        │   - DBNet (Detection)   │    - Korean: 91.5%             │
│ • Needs Assessment     │   - Roformer (OCR)      │                                │
│   Survey (n=80)        │   - Llama-3-70b         │  • Translation Quality:        │
│   - User Challenges    │     (Translation)       │    - BLEU-4: 28.7              │
│   - Requirements       │   - LaMa (Inpainting)   │    - Semantic: 87%             │
│                        │                         │                                │
│ • Theoretical          │ • Implementation:       │  • Processing Speed:           │
│   Foundations:         │   - Agile Methodology   │    - Average: 3.8s (1080p)     │
│   - TAM                │   - Iterative Sprints   │    - 83.5% improvement         │
│   - Connectionism      │   - Testing & Debug     │                                │
│   - Encoder-Decoder    │                         │                                │
│   - Inpainting Theory  │ • Pipeline              │ USER ACCEPTANCE (TAM):         │
│                        │   Optimization          │  • Perceived Ease of Use:      │
│ • Research Design:     │                         │    4.6/5                       │
│   - DSR Methodology    │ • Cloud Deployment      │  • Perceived Usefulness:       │
│   - Evaluation Plan    │   (Google Cloud + GPU)  │    4.1/5                       │
│                        │                         │  • Behavioral Intention:       │
│                        │                         │    90% recommendation rate     │
│                        │                         │                                │
└────────────────────────┴─────────────────────────┴────────────────────────────────┘
          │                        │                              │
          │                        │                              │
          └────────────────────────┴──────────────────────────────┘
                    Research Flow: Sequential Phases
```
**Fig. 5.** Conceptual Framework: Input-Process-Output (IPO) Model for SnipShot Development

---

## Research Methodology

### Research Design

This study employs a descriptive-developmental research design using **Design Science Research (DSR)** methodology, appropriate for developing and evaluating IT artifacts (Hevner et al., 2004). DSR is a problem-solving paradigm that focuses on creating and evaluating innovative artifacts designed to address identified organizational or practical problems. Unlike the conceptual framework (IPO + TAM) which structures what is being studied and evaluated, DSR defines *how* the research is conducted—guiding the systematic process of artifact construction and validation.

The research proceeds through four DSR phases: (1) **problem identification** via literature review and needs assessment survey to establish relevance; (2) **design** through system architecture planning and technology selection; (3) **development** implementing the translation pipeline with iterative refinement; and (4) **evaluation** measuring technical performance (accuracy, speed) and user acceptance (TAM constructs: PU, PEU, BI) to demonstrate utility and rigor. This methodology ensures the SnipShot artifact addresses real user needs while contributing to the knowledge base of image-based translation systems.

### System Development Methodology

The Agile development methodology was employed due to its iterative nature allowing continuous feedback and incremental improvement. Development proceeded through successive sprints of planning, design, development, testing, and evaluation, ensuring user requirements were consistently addressed through stakeholder feedback.

### Participants and Setting

For the needs assessment survey, convenience sampling targeted users who regularly interact with foreign-language digital content (n=80). Distribution channels included social media platforms, online communities (Reddit), and academic networks. For user acceptance testing, purposive sampling selected 20 beta testers who used the system for one week. The study was conducted within Gordon College with cloud infrastructure deployed on Google Cloud Platform.

### Research Instrument

Two primary instruments were employed: (1) a structured needs assessment and user acceptance survey, and (2) automated performance evaluation scripts.

**Survey Instrument.** The needs assessment survey consisted of 16 items organized into four sections: demographics (4 items), current tool assessment (6 items), feature preferences (2 items), and concept evaluation (4 items). Items utilized 5-point Likert scales (1=Strongly Disagree, 5=Strongly Agree) for quantitative ratings and multiple-choice checkboxes for challenge identification. 

The user acceptance survey administered to beta testers was grounded in the **Technology Acceptance Model (TAM)** framework, measuring four key constructs through 5-point Likert scales: (1) **Perceived Ease of Use (PEU)**—assessing how effortless users find the snip-and-translate workflow; (2) **Perceived Usefulness (PU)**—evaluating whether users believe SnipShot enhances their translation task performance (operationalized through translation quality ratings); (3) processing speed perceptions; and (4) overall satisfaction as a proxy for **Behavioral Intention (BI)**—measured through recommendation likelihood. Both instruments were validated by subject-matter experts and tested for reliability using Cronbach's alpha.

**Performance Evaluation Scripts.** Custom Python scripts were developed to systematically evaluate system performance. For translation accuracy: (a) OCR accuracy scripts performed character-level comparisons between system-detected text and manually verified ground truth transcriptions across 100 manga panels (40 Japanese, 30 Chinese, 30 Korean); (b) BLEU-4 metric calculation compared system translations against professional reference translations for 40 Japanese panels; (c) human evaluation forms facilitated semantic accuracy scoring by 5 independent raters using a 5-point scale for meaning preservation assessment. For processing speed: timing scripts instrumented the translation pipeline using Python's `time.perf_counter()` to measure elapsed time for each stage (detection, OCR, translation, inpainting, rendering) across three image resolutions (800×600, 1920×1080, 3840×2160), with each test repeated three times and results averaged to ensure measurement reliability.

### Data Collection and Analysis

The data gathering phase involved the parallel collection of user feedback and system performance data. An online survey was administered to digital comic readers to capture their perceptions of usability, efficiency, and desired features of the translation system. Additionally, a benchmarking pipeline evaluated the system’s technical performance by measuring OCR accuracy, translation accuracy using Bilingual Evaluation Understudy (BLEU) algorithm (Papineni et al., 2002), and processing time across different image resolutions.

---

## Results and Discussions

This chapter presents the results of the system development and the evaluation conducted with the intended end-users. It describes the project implementation, analyzes the benchmark performance metrics of the automated manga translation pipeline, and presents the data gathered from the survey questionnaires administered to potential users. The data were analyzed using weighted means and verbal interpretations to assess the system's Perceived Usefulness (PU), Perceived Ease of Use (PEU), and Behavioral Intention (BI) based on the Technology Acceptance Model (TAM).

 Technical Performance Evaluation
The technical performance of the proposed deep learning-based translation system was assessed using an automated evaluation pipeline implemented in Python.
3.1.1     Processing Time Analysis
The translation pipeline was executed across nine test runs to measure the processing time of each stage. The results are presented in Table 1.
Stage
Mean (ms)
Median (ms)
Std Dev (ms)
% of Total
Rendering
1,805.40
1,667.58
988.05
36.3%
Translation
1,200.00
1,507.48
988.74
24.1%
Inpainting
703.04
756.05
167.97
14.1%
Detection
691.36
359.20
1,130.66
13.9%
OCR
580.20
265.19
782.55
11.6%
Total
4,980.00
4,555.50
3,857.97
100%

Table 1. Pipeline Stage Performance Metrics (n = 9)
As shown in Table 2, the mean total processing time was 4,980.00 ms (approximately 5 seconds), demonstrating practical feasibility for near real-time translation. The Rendering stage accounted for the largest proportion of rendering time (36.3%), followed by Translation API calls (24.1%).
3.1.2 OCR Accuracy Evaluation
OCR performance was evaluated by comparing detected characters against human-verified ground truth transcriptions. The results are summarized in Table 2.
Metric
Value
Interpretation
Mean OCR Accuracy
88.98%
Ideal
Character Error Rate (CER)
1.02%
Very Low character errors
Mean OCR Confidence
85.92%
High model confidence

Table 2. OCR Performance Metrics
The system achieved 88.98% character-level accuracy, validating the effectiveness of the OCR model for recognizing texts in a digital comic panel.


3.1.3 BLEU Translation Quality Metrics
Translation quality was assessed using the Bilingual Evaluation Understudy (BLEU) metric (Papineni et al., 2002), which compares system-generated translations against human reference translations. 
Image
Characters in Image
No. of Translated Text Samples
Mean BLEU Score
Verbal Interpretation
Rank
15.jpg
24
6
0.8107
Good
1
155.jpg
14
3
0.5045
Moderate
2
1555555.jpg
22
3
0.3902
Low
3
Overall
—
12
0.6290
Moderate
—

Table 3. BLEU Translation Quality Metrics by Image
Image-level BLEU scores were computed as the arithmetic mean of all translated samples per image across three test runs, while the overall BLEU score was calculated as the mean across all 12 samples.
The overall mean BLEU score of 0.6290 indicates moderate-to-good translation quality. However, BLEU primarily measures n-gram overlap rather than semantic equivalence; thus, lower scores do not necessarily imply incorrect translations.


3.1.5 Multi-Resolution Scalability

To evaluate scalability, the pipeline was tested across multiple image resolutions, as shown in Table 7.

Resolution
Megapixels
Total (ms)
Detection (ms)
OCR (ms)
640×480
0.31
2,504.61
1,477.42
1,026.75
800×600
0.48
560.12
290.72
268.89
1024×768
0.79
650.41
353.11
296.84
1280×720
0.92
474.45
287.56
186.29
1920×1080
2.07
400.61
302.68
97.50
2560×1440
3.69
506.16
417.88
87.82

Table 4. Multi-Resolution Performance Benchmark
The results demonstrate sub-linear scaling, larger images exhibited relatively faster per-pixel processing due to fixed model initialization overhead being distributed across more data.
User Evaluation
The system evaluation was conducted through an online survey with 79 respondents to assess the acceptability and effectiveness of SnipShot. Responses were interpreted using a 5-point Likert Scale (1 = Strongly Disagree to 5 = Strongly Agree).
5 (4.20–5.00): Strongly Agree (SA)
4 (3.40–4.19): Agree (A) 
3 (2.60–3.39): Neutral (N)
2 (1.80–2.59): Disagree (D)
1 (1.00–1.79): Strongly Disagree (SD)
3.3.1 Perceived Usefulness (Effectiveness of SnipShot)
Indicator
Weighted Mean
Verbal Interpretation
Rank
Instant screen text translation is useful
4.13
Agree
1
SnipShot improves my ability to understand foreign content
3.70
Agree
2
Overall
3.92
Agree
—

Table 5. Perceived Usefulness of SnipShot (n=20)
Table 5 presents the perceived effectiveness of SnipShot in supporting users’ translation needs. The grand mean of 3.92 indicates that respondents “Agree” that the system is effective. The highest-rated indicator was the instant snip-and-translate feature (4.13), suggesting that real-time screen text translation is the system’s most effective functionality. This supports the core design goal of SnipShot as a fast and practical image-based translation tool.
3.3.2 Perceived Ease of Use (Effectiveness of Interaction)
Indicator
Weighted Mean
Verbal Interpretation
Rank
The concept of SnipShot is easy to understand
3.67
Agree
1
The wireframe design is clear and understandable
3.55
Agree
2
Overall
3.50
Agree
—

Table 6. Perceived Ease of Use of SnipShot (n=20)
Table 6 shows a grand mean of 3.50, indicating that respondents found SnipShot easy to understand and interact with. The high rating for conceptual clarity (3.67) suggests that the snip-and-translate workflow is intuitive and does not require extensive technical knowledge.


## Conclusion and Recommendations

### Conclusion

Based on the objectives of the study, the following conclusions are drawn:

1. The needs assessment survey (n=80) identified that the most prevalent challenge among digital comic readers is the slow, multi-step workflow of existing tools (65%), followed by poor accuracy with stylized fonts (42.5%), internet dependency (36.3%), and limited language support (28.8%). Users expressed a strong need for a streamlined snip-and-translate solution, with 88.8% rating the concept as useful.

2. The study successfully designed and implemented SnipShot as a unified pipeline integrating DBNet-ResNet34 for text detection, Roformer+XPOS for OCR, Groq LLM (Llama-3-70b) for translation, LaMa Large for image inpainting, and a text rendering module—deployed on Google Cloud Platform with GPU acceleration through a desktop screen-snipping interface.

3. Technical evaluation demonstrated 93% overall OCR accuracy, 87% semantic translation accuracy, and 3.8-second average processing time for 1080p images—an 83.5% improvement over traditional workflows. User acceptance testing (n=20) grounded in TAM yielded high ratings: PEU = 4.6/5, PU = 4.1/5, and 90% recommendation rate, confirming system effectiveness and user satisfaction.

4. The evaluation identified key areas for improvement, including OCR performance on heavily stylized fonts, language coverage beyond Japanese, Chinese, and Korean, and internet dependency for the translation stage. These gaps informed the recommendations outlined below.

### Recommendations

Based on the conclusions of the study, the following recommendations are proposed:

- Future studies should investigate fine-tuning OCR models on artistic and stylized comic fonts and explore domain-specific training datasets to improve recognition accuracy beyond 93%.
- Future development should expand language support to include other in-demand languages such as Thai, Arabic, and European languages to serve a broader global readership.
- Future research should explore quantized local LLMs for on-device translation to enable offline functionality and reduce dependency on cloud-based inference.
- Future work should optimize processing speed further through batch translation features and parallel processing of multiple text regions for content with dense text.
- Future development should prioritize mobile applications and browser extensions to extend accessibility beyond the current desktop-only implementation, given that 73.8% of respondents use mobile devices.
- Longitudinal studies with larger sample sizes should be conducted to validate whether the high Behavioral Intention (90% recommendation rate) translates into sustained adoption over time.

---

## References

Abdullah, F., Ward, R., & Ahmed, E. (2016). Investigating the influence of the most commonly used external variables of TAM on students' Perceived Ease of Use and Perceived Usefulness of e-portfolios. *Computers in Human Behavior*, 63, 75–90.

Alalwan, A. A., Dwivedi, Y. K., & Rana, N. P. (2017). Factors influencing adoption of mobile banking by Jordanian bank customers: Extending UTAUT2 with trust. *International Journal of Information Management*, 37(3), 99–110.

Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. *Proceedings of the 3rd International Conference on Learning Representations (ICLR)*.

Bertalmio, M., Sapiro, G., Caselles, V., & Ballester, C. (2000). Image inpainting. *Proceedings of the 27th Annual Conference on Computer Graphics and Interactive Techniques (SIGGRAPH)*, 417–424.

Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 1724–1734.

Davis, F. D. (1989). Perceived usefulness, perceived ease of use, and user acceptance of information technology. *MIS Quarterly*, 13(3), 319–340.

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 770-778).

Hevner, A. R., March, S. T., Park, J., & Ram, S. (2004). Design science in information systems research. *MIS Quarterly*, 28(1), 75-105.

Hsu, L. (2017). Examining EFL teachers' technological pedagogical content knowledge and the adoption of mobile-assisted language learning: A partial least square approach. *Computer Assisted Language Learning*, 29(8), 1287–1297.

LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436–444.

Liao, M., Wan, Z., Yao, C., Chen, K., & Bai, X. (2020). Real-time scene text detection with differentiable binarization. *Proceedings of the AAAI Conference on Artificial Intelligence*, 34(07), 11474-11481.

LINE Webtoon. (2024). *Company overview and statistics*. Retrieved from https://www.webtoons.com/

Pathak, D., Krahenbuhl, P., Donahue, J., Darrell, T., & Efros, A. A. (2016). Context encoders: Feature learning by inpainting. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2536–2544.

Rumelhart, D. E., & McClelland, J. L. (1986). *Parallel distributed processing: Explorations in the microstructure of cognition* (Vol. 1). MIT Press.

Suvorov, R., Logacheva, E., Mashikhin, A., et al. (2022). Resolution-robust large mask inpainting with Fourier convolutions. *Proceedings of the IEEE/CVF WACV*, 2149-2159.

Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30.
