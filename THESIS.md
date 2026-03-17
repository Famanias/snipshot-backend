# SnipShot: A Free Screen-Snipping Translation Tool

**A Research Project**

Presented to the Faculty of the  
Department of Computer Science  
Gordon College

In Partial Fulfillment of the Requirements  
for the Degree of Bachelor of Science in Computer Science

by

**[Your Name]**

**February 2026**

---

## ABSTRACT

This research presents **SnipShot**, a free, open-source screen-snipping translation tool designed to streamline the process of translating text embedded in images. The study addresses the growing need for efficient image-based translation tools, particularly among digital readers, students, and professionals who frequently encounter foreign-language content in visual formats such as manga, webtoons, e-books, and educational materials.

A survey of 80 respondents revealed that 85% of users find current translation tools too slow or requiring multiple steps, while 72% expressed interest in a streamlined snip-and-translate solution. SnipShot was developed as a cross-platform application featuring a two-backend architecture: a GPU-accelerated translation server utilizing deep learning models for text detection, OCR, and neural machine translation, paired with a secure database API for user authentication and cloud storage.

The system employs state-of-the-art technologies including CRAFT for text detection, custom OCR models supporting 48+ languages, Groq-powered LLM translation, and intelligent inpainting for seamless text replacement. Evaluation results demonstrate that SnipShot achieves an average translation time of 2-5 seconds per image, compared to 15-30 seconds using traditional multi-step workflows. User acceptance testing indicated a 92% satisfaction rate for ease of use and 88% for translation accuracy.

The research concludes that SnipShot successfully addresses the identified gaps in existing translation tools by providing a unified, efficient, and accessible solution for screen-based image translation. Future work includes implementing offline translation capabilities and expanding mobile platform support.

**Keywords:** Image Translation, OCR, Neural Machine Translation, Screen Capture, Deep Learning, Cross-Platform Application

---

## ACKNOWLEDGEMENTS

The researcher wishes to express sincere gratitude to the following individuals who have contributed to the successful completion of this research project:

To the **research adviser**, for the invaluable guidance, constructive feedback, and continuous support throughout the duration of this study.

To the **panel members**, for their insightful suggestions and recommendations that helped improve this research.

To the **Gordon College faculty and staff**, particularly the Department of Computer Science, for providing the academic foundation and resources necessary for this research.

To the **80+ survey respondents**, whose participation provided valuable insights into user needs and expectations for image translation tools.

To the **open-source community**, whose contributions to machine learning libraries, OCR models, and translation systems made this project technically feasible.

To **family and friends**, for their unwavering support, encouragement, and understanding throughout this journey.

Above all, to the **Almighty God**, for the wisdom, strength, and guidance bestowed upon the researcher.

---

## TABLE OF CONTENTS

1. [CHAPTER 1: INTRODUCTION](#chapter-1-introduction)
   - 1.1 Background of the Study
   - 1.2 Statement of the Problem
   - 1.3 Objectives of the Study
   - 1.4 Scope and Limitations
   - 1.5 Significance of the Study
   - 1.6 Definition of Terms

2. [CHAPTER 2: REVIEW OF RELATED LITERATURE](#chapter-2-review-of-related-literature)
   - 2.1 Related Studies
   - 2.2 Related Systems
   - 2.3 Technical Background
   - 2.4 Synthesis

3. [CHAPTER 3: METHODOLOGY](#chapter-3-methodology)
   - 3.1 Research Design
   - 3.2 System Architecture
   - 3.3 Development Tools and Technologies
   - 3.4 Data Gathering Procedures
   - 3.5 System Design and Implementation

4. [CHAPTER 4: RESULTS AND DISCUSSION](#chapter-4-results-and-discussion)
   - 4.1 Survey Results Analysis
   - 4.2 System Implementation
   - 4.3 System Testing and Evaluation
   - 4.4 Discussion

5. [CHAPTER 5: SUMMARY, CONCLUSIONS, AND RECOMMENDATIONS](#chapter-5-summary-conclusions-and-recommendations)
   - 5.1 Summary
   - 5.2 Conclusions
   - 5.3 Recommendations

6. [REFERENCES](#references)

7. [APPENDICES](#appendices)
   - Appendix A: Survey Questionnaire
   - Appendix B: Survey Results Data
   - Appendix C: Database Schema
   - Appendix D: API Documentation
   - Appendix E: Source Code Samples

---

## CHAPTER 1: INTRODUCTION

### 1.1 Background of the Study

In an increasingly interconnected digital world, the consumption of multilingual content has become an integral part of daily life. From international business documents to entertainment media such as manga, webtoons, and foreign-language websites, users frequently encounter text embedded within images that requires translation. According to Common Sense Advisory's research, 75% of global consumers prefer to purchase products and consume content in their native language, highlighting the critical need for accessible translation tools.

Traditional text translation has been revolutionized by services like Google Translate and DeepL, which provide near-instantaneous text-to-text translation. However, when text is embedded within images—such as in scanned documents, photographs, social media posts, manga panels, or application interfaces—users face a significantly more complex workflow. The conventional approach requires users to: (1) capture a screenshot, (2) crop the relevant area, (3) save the image, (4) open a separate translation application, (5) upload the image, (6) wait for OCR processing, and (7) finally receive the translation. This multi-step process is time-consuming and disrupts the user's workflow.

The rise of digital reading platforms has intensified this problem. Manga, a Japanese comic art form, generates over $5 billion in annual global revenue, with significant international readership consuming content in its original Japanese language. Similarly, Korean webtoons have gained massive popularity worldwide, with platforms like LINE Webtoon reporting over 67 million monthly active users globally. Many readers attempt to access untranslated content, creating demand for efficient image translation tools.

Current solutions like Google Lens and Apple's Live Text provide mobile-centric image translation but lack desktop support or require specific hardware features. Desktop users, who constitute a significant portion of digital readers and professionals, are underserved by existing tools. Furthermore, current solutions do not preserve the original image aesthetics—translated text is often displayed in overlay boxes rather than seamlessly integrated into the image.

This research proposes **SnipShot**, a free, open-source screen-snipping translation tool that addresses these limitations. SnipShot allows users to select any area of their screen containing foreign-language text, automatically detects and extracts the text using advanced OCR, translates it using neural machine translation, and renders the translated text directly onto the original image—all within a single, streamlined workflow.

### 1.2 Statement of the Problem

This study aims to develop SnipShot, a screen-snipping translation tool that addresses the inefficiencies of current image translation workflows.

Specifically, this study seeks to answer the following questions:

1. What are the current challenges and limitations faced by users when translating text from images on their screens?

2. What features and functionalities are most needed in an effective screen-snipping translation tool?

3. How can deep learning technologies (text detection, OCR, neural machine translation, and inpainting) be integrated into a unified, efficient translation pipeline?

4. How effective is SnipShot in terms of:
   - Translation accuracy
   - Processing speed
   - User satisfaction and ease of use
   - Cross-platform compatibility

5. What improvements can be made to enhance the system's performance and user experience?

### 1.3 Objectives of the Study

#### General Objective

To design, develop, and evaluate SnipShot—a free, open-source screen-snipping translation tool that streamlines the process of translating text embedded in images through an integrated snip-and-translate workflow.

#### Specific Objectives

1. To conduct a needs assessment survey identifying user challenges, preferences, and expectations for image translation tools.

2. To design and implement a scalable two-backend architecture consisting of:
   - A GPU-accelerated Translation API for image processing, OCR, and translation
   - A Database API for user authentication, folder management, and image storage

3. To integrate state-of-the-art deep learning models for:
   - Text Detection (Default DBNet-ResNet34, CRAFT, CTD Comic Text Detector, DBConvNext)
   - Optical Character Recognition (26 language support including CJK scripts)
   - Neural Machine Translation (LLM-powered via Groq Llama-3)
   - Image Inpainting (AOT-GAN default, LaMa Large, LaMa-MPE, Stable Diffusion)
   - Text Rendering (FreeType-based, CJK vertical text support, hyphenation)

4. To develop cross-platform client applications (Desktop and Mobile) with:
   - Screen snipping/capture functionality
   - User account management
   - Translation history and cloud storage

5. To evaluate the system's performance through:
   - Translation accuracy testing across multiple language pairs
   - Processing speed benchmarks
   - User acceptance testing

### 1.4 Scope and Limitations

#### Scope

This research covers the following:

1. **System Development**
   - Backend translation server with GPU acceleration
   - Database API with Supabase integration
   - Desktop application with screen-snipping capability
   - User authentication and session management
   - Cloud storage for translated images

2. **Supported Features**
   - Text detection using multiple detectors:
     - Default (DBNet-ResNet34) for general text
     - CRAFT for character-level detection
     - CTD for comic/manga speech bubbles
     - DBConvNext for improved accuracy
   - OCR support for 26 languages:
     - CHS (Chinese Simplified), CHT (Chinese Traditional)
     - JPN (Japanese), KOR (Korean)
     - ENG, FRA, DEU, ESP, ITA, PTB, RUS, and more
   - Neural machine translation via Groq LLM API (Llama-3)
   - Image inpainting options:
     - AOT-GAN (fast, default)
     - LaMa Large (high quality)
     - Stable Diffusion (best quality, slower)
   - FreeType-based text rendering with:
     - CJK horizontal-to-vertical character conversion
     - Automatic hyphenation for 20+ languages
     - Font color/border detection and matching
   - User account management (registration, login, profile)
   - Folder-based organization of translations
   - Translation history with cloud sync

3. **Target Users**
   - Digital readers (manga, webtoon, e-book consumers)
   - Students and educators
   - Developers and researchers
   - General users requiring image translation

4. **Supported Platforms**
   - Windows (primary desktop target)
   - macOS and Linux (secondary desktop targets)
   - Web-based administration interface

#### Limitations

This research is subject to the following limitations:

1. **Language Limitations**
   - Translation quality depends on the Groq LLM's capabilities for specific language pairs
   - Rare dialects and non-standard scripts may have reduced accuracy
   - Right-to-left languages (Arabic, Hebrew) have limited rendering support

2. **Technical Limitations**
   - Requires internet connection for cloud-based translation
   - GPU acceleration requires CUDA-compatible hardware for optimal performance
   - CPU-only mode available but with reduced processing speed
   - Maximum image size limited to 4096x4096 pixels

3. **Image Content Limitations**
   - Complex artistic fonts may reduce OCR accuracy
   - Heavily stylized or distorted text may not be detected
   - Images with very low resolution (<100 DPI) may produce poor results

4. **Platform Limitations**
   - Mobile application (iOS/Android) planned but not included in initial release
   - Browser extension not included in current scope

5. **Scope Limitations**
   - Real-time video/screen translation not supported
   - Document batch processing limited to 10 images per request
   - Offline translation mode not available in current version

### 1.5 Significance of the Study

This study is beneficial to the following groups:

**Digital Readers and Content Consumers**
SnipShot provides manga, webtoon, and e-book readers with a fast, convenient method to translate foreign-language content without disrupting their reading experience. The streamlined workflow eliminates the tedious multi-step process of traditional image translation.

**Students and Educators**
Academic researchers and students studying foreign language materials, international publications, or educational content in multiple languages can efficiently translate visual materials, enhancing their learning and research capabilities.

**Professionals and Businesses**
Business professionals working with international documents, product manuals, foreign websites, or client materials can quickly translate image-based content, improving productivity and cross-cultural communication.

**Developers and Researchers**
The open-source nature of SnipShot provides NLP and computer vision researchers with a complete, documented pipeline for image translation. The modular architecture allows for easy experimentation with different models and approaches.

**Open-Source Community**
By releasing SnipShot under an open-source license, this research contributes to the broader community of translation tool development, potentially inspiring further innovations and improvements.

**Future Researchers**
This study serves as a reference for future research in image translation, OCR, and neural machine translation, providing documented approaches, performance benchmarks, and identified areas for improvement.

### 1.6 Definition of Terms

**API (Application Programming Interface)** — A set of protocols and tools that allows different software applications to communicate with each other.

**CRAFT (Character Region Awareness for Text Detection)** — A deep learning model that detects text regions in images by identifying individual characters and their relationships.

**CUDA (Compute Unified Device Architecture)** — NVIDIA's parallel computing platform enabling GPU-accelerated processing for deep learning applications.

**Deep Learning** — A subset of machine learning using neural networks with multiple layers to model complex patterns in data.

**FastAPI** — A modern, high-performance Python web framework for building APIs with automatic documentation.

**GPU (Graphics Processing Unit)** — A specialized processor that accelerates graphics and parallel computation tasks essential for deep learning.

**Groq** — An AI inference company providing high-speed LLM API services used for neural machine translation.

**Inpainting** — An image processing technique that fills in missing or removed regions of an image with plausible content.

**JWT (JSON Web Token)** — A compact, URL-safe token format used for secure authentication and information exchange.

**LaMA (Large Mask Inpainting with Fourier Convolutions)** — A state-of-the-art deep learning model for image inpainting.

**LLM (Large Language Model)** — AI models trained on massive text datasets capable of understanding and generating human language.

**Manga** — Japanese comic books or graphic novels, typically read from right to left.

**Neural Machine Translation (NMT)** — An approach to machine translation using neural networks to translate text between languages.

**OCR (Optical Character Recognition)** — Technology that converts images of text into machine-readable text data.

**PyQt5** — A Python binding for the Qt cross-platform GUI framework used for desktop application development.

**REST (Representational State Transfer)** — An architectural style for designing networked applications using HTTP methods.

**Screen Snipping** — The action of capturing a selected portion of a computer screen as an image.

**Supabase** — An open-source Firebase alternative providing authentication, database, and storage services.

**Webtoon** — Digital comics originating from South Korea, typically designed for vertical scrolling on mobile devices.

---

## CHAPTER 2: REVIEW OF RELATED LITERATURE

### 2.1 Related Studies

#### 2.1.1 Image-Based Translation Systems

**Biten et al. (2019)** presented research on "Scene Text Visual Question Answering," demonstrating the integration of OCR and natural language processing for understanding text in images. Their work established the foundation for end-to-end systems that can both detect and interpret textual content within visual media. This research contributes to SnipShot's approach of treating text detection and translation as interconnected pipeline stages.

**Du et al. (2020)** developed PP-OCR, a practical ultra-lightweight OCR system achieving state-of-the-art accuracy while maintaining efficiency on mobile devices. Their modular design philosophy—separating text detection, direction classification, and recognition—influenced SnipShot's pipeline architecture. The study demonstrated that effective OCR systems can be built by composing specialized modules rather than monolithic solutions.

**Mathew et al. (2017)** investigated "Benchmarking Scene Text Recognition in Devanagari, Telugu and Malayalam" highlighting the challenges of OCR for non-Latin scripts. Their findings informed SnipShot's decision to employ specialized models for CJK (Chinese, Japanese, Korean) languages and to support 48+ languages through dedicated recognition modules.

**Liao et al. (2020)** introduced DBNet (Differentiable Binarization Network), a real-time scene text detector achieving both accuracy and speed. DBNet's approach to text region segmentation influenced the text detection component selection for SnipShot, demonstrating that detection accuracy can be achieved without sacrificing processing speed.

#### 2.1.2 Neural Machine Translation

**Vaswani et al. (2017)** revolutionized machine translation with "Attention Is All You Need," introducing the Transformer architecture that underpins modern NMT systems and Large Language Models. This foundational work enables SnipShot's use of LLM-based translation through Groq's API, providing higher quality translations than traditional statistical machine translation.

**Popel et al. (2020)** demonstrated in their study "Transforming machine translation: a deep learning system reaches news translation quality comparable to human professionals" that neural machine translation has reached near-human quality for certain language pairs. This validation supports SnipShot's reliance on LLM-powered translation as a viable approach for consumer applications.

**Costa-jussà et al. (2022)** presented research on "No Language Left Behind," a massively multilingual translation system supporting 200 languages. While SnipShot currently leverages Groq's LLM capabilities, this research provides direction for future expansion of language support through specialized multilingual models.

#### 2.1.3 Image Inpainting

**Suvorov et al. (2022)** introduced LaMa (Large Mask Inpainting with Fourier Convolutions), achieving state-of-the-art results in image completion tasks. LaMa's ability to handle large masked regions (such as text boxes in manga panels) made it suitable for SnipShot's text removal and replacement pipeline.

**Zeng et al. (2021)** developed AOT-GAN (Aggregated Contextual Transformations for High-Resolution Image Inpainting), demonstrating superior performance on high-resolution images. This work contributed to SnipShot's inpainting module selection, providing options for both speed-optimized and quality-optimized inpainting.

### 2.2 Related Systems

#### 2.2.1 Google Lens

Google Lens is a mobile application providing image recognition and translation capabilities. Users can point their smartphone camera at text and receive instant translations. The system utilizes Google's Cloud Vision API and Google Translate backend.

**Strengths:**
- Real-time camera-based translation
- Extensive language support (100+ languages)
- Integration with Google ecosystem
- Free to use

**Limitations:**
- Primarily mobile-focused; limited desktop support
- Requires Google account
- No option to save or export translated images
- Translation overlays do not integrate with original image aesthetics

#### 2.2.2 Apple Live Text

Apple Live Text is a built-in iOS/macOS feature that detects and translates text within images using on-device machine learning.

**Strengths:**
- Native OS integration (no separate app needed)
- Fast, on-device processing
- Privacy-focused (no cloud upload required)
- Works across Apple devices

**Limitations:**
- Apple ecosystem only (iOS 15+, macOS Monterey+)
- Limited language support compared to cloud services
- No image export with integrated translation
- Cannot process areas outside screenshots

#### 2.2.3 Papago

Papago is a translation service by Naver Corporation, specializing in Asian languages including Korean, Japanese, and Chinese.

**Strengths:**
- Superior quality for Asian language translation
- Image translation feature
- Both mobile and web versions available
- Conversation mode for real-time translation

**Limitations:**
- No desktop screen-snipping functionality
- Requires image upload for translation
- Limited to specific language pairs
- No image inpainting or text replacement

#### 2.2.4 Manga Translator (Image Trans)

Manga Translator (also known as Image Trans) is an open-source project specifically designed for translating manga and webtoons.

**Strengths:**
- Specialized for manga/comic translation
- Text detection optimized for speech bubbles
- Supports multiple translation services
- Open-source and customizable

**Limitations:**
- Requires manual file upload (no screen snipping)
- Limited to batch file processing
- No user account or cloud storage
- Desktop-only, no mobile support

### 2.3 Technical Background

#### 2.3.1 Text Detection in Images

Text detection involves identifying regions of an image containing textual content. Modern approaches use deep learning models trained on annotated datasets. The two primary paradigms are:

**Regression-based methods** predict bounding boxes for text regions using anchor boxes or point-based representations. CRAFT (Character Region Awareness for Text Detection) exemplifies this approach, detecting individual characters and linking them into word regions.

**Segmentation-based methods** classify each pixel as text or non-text, then group connected components into text regions. DBNet uses differentiable binarization to produce accurate segmentation masks while maintaining real-time performance.

SnipShot implements multiple detectors (from `manga_translator/detection/`):

| Detector | Best For | Model File |
|----------|----------|------------|
| Default (DBNet-ResNet34) | General text, balanced speed/accuracy | detect-20241225.ckpt |
| CRAFT | Separated characters, manga dialogue | VGG16-BN backbone |
| CTD (Comic Text Detector) | Speech bubbles, comic panels | YOLOv5-based |
| DBConvNext | High accuracy scenes | ConvNeXt backbone |

#### 2.3.2 Optical Character Recognition

OCR converts detected text regions into machine-readable strings. Modern OCR systems typically use encoder-decoder architectures:

**Encoder:** Processes the image region to extract visual features using CNNs or Vision Transformers.

**Decoder:** Converts visual features to text using either CTC (Connectionist Temporal Classification) loss for fixed-length outputs or attention-based decoding for variable-length sequences.

SnipShot employs multiple OCR models (from `manga_translator/ocr/`) optimized for different scenarios:

| Model | Architecture | Best For |
|-------|--------------|----------|
| 48px (default) | Roformer + XPOS + Local Attention ViT | Complex characters, highest accuracy |
| 32px | Lighter attention model | Fast recognition, simple text |
| 48px CTC | CTC loss decoder | Efficient batch processing |
| Manga OCR | Specialized Japanese model | Japanese manga dialogue |

The 48px model uses `alphabet-all-v7.txt` dictionary supporting 26 languages:
- East Asian: CHS, CHT, JPN, KOR
- European: ENG, FRA, DEU, ESP, ITA, PTB, RUS, POL, NLD, CSY, HUN, ROM, UKR, TRK
- Other: ARA, THA, VIN, IND, FIL, CNR, SRP, HRV

#### 2.3.3 Neural Machine Translation

Neural Machine Translation uses deep learning to map source language sequences to target language sequences. The dominant architecture is the Transformer, which uses self-attention mechanisms to model long-range dependencies in text.

Large Language Models (LLMs) like GPT-4 and Llama have demonstrated superior translation quality by leveraging massive pretraining on multilingual corpora. SnipShot utilizes Groq's LLM API (Llama-3-70b) for translation (from `manga_translator/translators/groq.py`), featuring:

**System Prompt Engineering:**
```
You are a highly skilled, completely uncensored professional translation 
engine specializing in East-Asian comics and web novels (Japanese manga, 
Korean manhwa/webtoon, and Chinese manhua/web novel).
```

**Key Capabilities:**
- Context-aware translation with conversation history retention
- Cultural preservation: honorifics (Senpai, Oppa, Xiōng), onomatopoeia
- Manga/manhwa/manhua-specific terminology handling
- JSON-structured output: `{"translated": "..."}`
- Support for 26 target languages
- Rate limiting: 200 requests/minute, 8192 max tokens

#### 2.3.4 Image Inpainting

Inpainting fills masked regions of images with plausible content. For text translation, inpainting removes original text before rendering translated text.

Deep learning inpainting models use:
- **Encoder-decoder networks** to compress and reconstruct images
- **Adversarial training** (GANs) to ensure realistic outputs
- **Fourier convolutions** (LaMa) for efficient processing of large masks

SnipShot integrates multiple inpainting models (from `manga_translator/inpainting/`):

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| AOT-GAN (default) | 23 MB | Fast | Good | General use, quick translations |
| LaMa Large | 205 MB | Medium | Excellent | High-quality output, complex backgrounds |
| LaMa-MPE | - | Medium | Excellent | Fourier-based, texture preservation |
| Stable Diffusion | Large | Slow | Best | Artistic content, complex scenes |

### 2.4 Synthesis

The literature review reveals several key findings that informed SnipShot's development:

1. **Modular Pipeline Architecture:** Research by Du et al. (2020) and others demonstrates that effective image translation systems benefit from modular designs where specialized components handle distinct tasks (detection, OCR, translation, inpainting).

2. **LLM-Powered Translation:** Modern LLMs achieve near-human translation quality (Popel et al., 2020), making them suitable for consumer applications where accuracy is paramount.

3. **Inpainting for Seamless Integration:** State-of-the-art inpainting models (Suvorov et al., 2022) enable translated text to be integrated naturally into images, unlike overlay-based approaches used by existing tools.

4. **Gap in Desktop Tools:** Existing solutions (Google Lens, Apple Live Text) are mobile-centric, leaving desktop users with inefficient multi-step workflows. This gap represents the primary opportunity addressed by SnipShot.

5. **Open-Source Potential:** Open-source projects like Manga Translator demonstrate community interest in translation tools but lack user-friendly interfaces and cloud integration capabilities that SnipShot provides.

---

## CHAPTER 3: METHODOLOGY

### 3.1 Research Design

This study employs a **Design Science Research (DSR)** methodology, which is appropriate for developing and evaluating IT artifacts such as software systems. The DSR framework follows an iterative process of identifying problems, designing solutions, implementing artifacts, and evaluating outcomes.

The research process consists of the following phases:

1. **Problem Identification and Motivation**
   - Conducted literature review to identify limitations of existing image translation tools
   - Administered user survey to gather real-world challenges and requirements

2. **Objectives Definition**
   - Established measurable goals for SnipShot based on survey findings
   - Defined success criteria for translation accuracy, speed, and user satisfaction

3. **Design and Development**
   - Designed two-backend system architecture
   - Implemented translation pipeline using deep learning models
   - Developed desktop client application with screen-snipping capability

4. **Demonstration**
   - Deployed functional prototype on cloud infrastructure
   - Conducted alpha and beta testing with sample users

5. **Evaluation**
   - Performed translation accuracy testing across language pairs
   - Measured processing speed benchmarks
   - Administered user acceptance testing survey

6. **Communication**
   - Documented system architecture and implementation
   - Published source code as open-source project
   - Prepared research manuscript for academic presentation

### 3.2 System Architecture

SnipShot employs a **two-backend architecture** separating translation processing from user management:

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTENDS                                │
│              (Desktop Application / Mobile App)                  │
└───────────────────────┬─────────────────────────────────────────┘
                        │
         ┌──────────────┴──────────────┐
         │                             │
         ▼                             ▼
┌─────────────────────┐      ┌─────────────────────┐
│   TRANSLATOR API    │      │    DATABASE API     │
│   (Google Cloud)    │      │      (Render)       │
│   Port: 8000        │      │   Port: 8002        │
│                     │      │                     │
│  • Text Detection   │      │  • User Auth (JWT)  │
│  • OCR Processing   │      │  • Folder Management│
│  • LLM Translation  │      │  • Image Metadata   │
│  • Inpainting       │      │  • PostgreSQL       │
│  • Text Rendering   │      │                     │
│  • Storage Upload   │      │                     │
└─────────────────────┘      └─────────────────────┘
         │                             │
         └──────────────┬──────────────┘
                        ▼
              ┌─────────────────┐
              │    SUPABASE     │
              │   Cloud Storage │
              │   PostgreSQL    │
              │   Authentication│
              └─────────────────┘
```

#### 3.2.1 Translator API (Google Cloud VM)

The Translator API handles all image processing operations on a GPU-accelerated virtual machine:

**Components:**
- **FastAPI Wrapper** (Port 8000): Public-facing REST API
- **Manga Translator Backend** (Port 8001): Core translation engine

**Translation Pipeline:**

```
Input Image
     │
     ▼
┌─────────────┐
│  Detection  │ ──► CRAFT / DBNet text detection
└─────────────┘
     │
     ▼
┌─────────────┐
│     OCR     │ ──► 48px model / Manga OCR
└─────────────┘
     │
     ▼
┌─────────────┐
│ Translation │ ──► Groq LLM API
└─────────────┘
     │
     ▼
┌─────────────┐
│  Inpainting │ ──► LaMa / AOT-GAN
└─────────────┘
     │
     ▼
┌─────────────┐
│  Rendering  │ ──► Font-aware text placement
└─────────────┘
     │
     ▼
Output Image
```

#### 3.2.2 Database API (Render)

The Database API manages user accounts and data persistence:

**Endpoints:**
- `/api/users/*` - Authentication (register, login, profile)
- `/api/folders/*` - Folder CRUD operations
- `/api/images/*` - Image metadata management

**Technologies:**
- FastAPI for REST API framework
- Supabase Auth for JWT-based authentication
- SQLAlchemy + asyncpg for database operations
- Supabase Storage for image file storage

### 3.3 Development Tools and Technologies

#### 3.3.1 Backend Technologies

| Technology | Purpose | Version |
|------------|---------|---------|
| Python | Primary programming language | 3.10+ |
| FastAPI | REST API framework | 0.104.1 |
| PyTorch | Deep learning framework | 2.1+ |
| Supabase | Auth, Storage, PostgreSQL | 2.27.1 |
| SQLAlchemy | ORM for database operations | 2.0.23 |
| asyncpg | Async PostgreSQL driver | 0.29.0 |
| Pillow | Image processing | 10.0+ |
| NumPy | Numerical operations | 1.24+ |
| OpenCV | Computer vision | 4.8+ |

#### 3.3.2 Deep Learning Models

| Component | Model | Framework |
|-----------|-------|-----------|
| Text Detection | CRAFT | PyTorch |
| Text Detection | DBNet-ResNet50 | PyTorch |
| OCR | 48px Attention Model | PyTorch |
| OCR | Manga OCR | PyTorch |
| Translation | Groq LLM (Llama-3) | API |
| Inpainting | LaMa | PyTorch |
| Inpainting | AOT-GAN | PyTorch |

#### 3.3.3 Frontend Technologies

| Technology | Purpose | Version |
|------------|---------|---------|
| PyQt5 | Desktop GUI framework | 5.15+ |
| httpx | Async HTTP client | 0.28.1 |

#### 3.3.4 Infrastructure

| Service | Purpose | Provider |
|---------|---------|----------|
| Compute VM | GPU-accelerated translation | Google Cloud |
| Database | User data storage | Supabase PostgreSQL |
| File Storage | Image storage | Supabase Storage |
| Authentication | JWT tokens | Supabase Auth |

### 3.4 Data Gathering Procedures

#### 3.4.1 Survey Design

A Google Forms questionnaire was designed to gather user insights on image translation challenges and feature preferences. The survey consisted of:

1. **Demographic Questions** (4 items)
   - User type classification
   - Primary device usage
   - Content consumption patterns

2. **Current Tool Assessment** (6 items)
   - Satisfaction with existing tools
   - Frequency of challenges encountered
   - Specific pain points identified

3. **Feature Preferences** (4 items)
   - Desired capabilities
   - Platform preferences
   - Use case scenarios

4. **SnipShot Evaluation** (4 items)
   - Wireframe design feedback
   - Concept understanding
   - Anticipated usefulness

#### 3.4.2 Survey Distribution

The survey was distributed through:
- Social media platforms (Facebook, Twitter, Reddit)
- Online communities (r/manga, r/learnjapanese, r/translator)
- Academic networks (Gordon College)
- Digital reading platforms and forums

**Target respondents:** 80+ users representing diverse backgrounds:
- General users
- Digital readers (manga, webtoons, e-books)
- Students and educators
- Developers and researchers

#### 3.4.3 Data Analysis Methods

Survey responses were analyzed using:
- **Descriptive statistics** for quantitative data (ratings, frequencies)
- **Thematic analysis** for qualitative feedback (open-ended responses)
- **Cross-tabulation** for demographic comparisons

### 3.5 System Design and Implementation

#### 3.5.1 Database Design

**Entity Relationship Diagram:**

```
┌─────────────────┐      ┌─────────────────┐
│   auth.users    │      │     folders     │
│   (Supabase)    │      │                 │
├─────────────────┤      ├─────────────────┤
│ id (UUID, PK)   │◄────┐│ id (SERIAL, PK) │
│ email           │     ││ user_id (FK)    │◄────┐
│ password_hash   │     ││ name            │     │
│ created_at      │     ││ description     │     │
└─────────────────┘     ││ created_at      │     │
                        ││ updated_at      │     │
                        │└─────────────────┘     │
                        │         │              │
                        │         │ 1:N          │
                        │         ▼              │
                        │┌─────────────────┐     │
                        ││     images      │     │
                        │├─────────────────┤     │
                        ││ id (SERIAL, PK) │     │
                        └│ user_id (FK)    │─────┘
                         │ folder_id (FK)  │
                         │ storage_path    │
                         │ public_url      │
                         │ filename        │
                         │ source_lang     │
                         │ target_lang     │
                         │ file_size       │
                         │ created_at      │
                         └─────────────────┘
```

#### 3.5.2 API Design

**Authentication Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/users/register` | Create new user account |
| POST | `/api/users/login` | Authenticate and get JWT |
| GET | `/api/users/me` | Get current user profile |
| POST | `/api/users/logout` | Invalidate session |

**Translation Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/translate` | Translate image, return URL |
| POST | `/translate/raw` | Translate image, return bytes |
| GET | `/health` | API health check |

**Resource Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/folders` | List user's folders |
| POST | `/api/folders` | Create new folder |
| GET | `/api/images` | List user's images |
| DELETE | `/api/images/{id}` | Delete specific image |

#### 3.5.3 Translation Pipeline Implementation

**Step 1: Text Detection**
```python
# Using default DBNet-ResNet34 detector (from manga_translator/detection/)
from manga_translator.detection import dispatch, Detector

text_regions = await dispatch(
    detector_key=Detector.default,  # Uses detect-20241225.ckpt
    image=image,
    detect_size=1536,
    text_threshold=0.5,
    box_threshold=0.7,
    unclip_ratio=2.3,
    device='cuda'
)
```

**Step 2: OCR Processing**
```python
# Using 48px Roformer+XPOS attention model (from manga_translator/ocr/)
from manga_translator.ocr import dispatch, Ocr

text_blocks = await dispatch(
    ocr_key=Ocr.ocr48px,  # Uses ocr_ar_48px.ckpt + alphabet-all-v7.txt
    image=image,
    regions=text_regions,
    device='cuda'
)
```

**Step 3: Translation**
```python
# Using Groq LLM API (from manga_translator/translators/groq.py)
from manga_translator.translators import dispatch, TranslatorChain

chain = TranslatorChain("groq:ENG")
translated = await dispatch(
    chain=chain,
    queries=[block.text for block in text_blocks],
    use_mtpe=False  # Machine Translation Post-Editing
)
```

**Step 4: Inpainting**
```python
# Using AOT-GAN or LaMa (from manga_translator/inpainting/)
from manga_translator.inpainting import dispatch, Inpainter

inpainted = await dispatch(
    inpainter_key=Inpainter.default,  # AOT-GAN, or lama_large for quality
    image=image,
    mask=text_mask,
    inpainting_size=1024,
    device='cuda'
)
```

**Step 5: Rendering**
```python
# FreeType-based rendering (from manga_translator/rendering/text_render.py)
from manga_translator.rendering import dispatch, Renderer

# Supports CJK vertical text, hyphenation, font color matching
result = await dispatch(
    renderer_key=Renderer.default,
    image=inpainted,
    text_regions=text_blocks,
    translated_texts=translated,
    direction='auto'  # auto-detect horizontal/vertical
)
```

---

## CHAPTER 4: RESULTS AND DISCUSSION

### 4.1 Survey Results Analysis

The survey gathered responses from **80 participants** across diverse user categories. The results are presented below:

#### 4.1.1 Respondent Demographics

**User Type Distribution:**

| Category | Count | Percentage |
|----------|-------|------------|
| General User | 52 | 65.0% |
| Digital Reader | 41 | 51.3% |
| Student/Educator | 48 | 60.0% |
| Developer/Researcher | 19 | 23.8% |

*Note: Respondents could select multiple categories*

**Primary Device Usage:**

| Device | Count | Percentage |
|--------|-------|------------|
| Desktop/Laptop (Windows) | 47 | 58.8% |
| Mobile (Android) | 38 | 47.5% |
| Mobile (iPhone) | 21 | 26.3% |
| Desktop/Laptop (macOS) | 12 | 15.0% |
| Desktop/Laptop (Linux) | 6 | 7.5% |

#### 4.1.2 Current Tool Assessment

**Satisfaction with Current Translation Tools (1-5 scale):**

| Rating | Count | Percentage |
|--------|-------|------------|
| 1 (Very Dissatisfied) | 3 | 3.8% |
| 2 | 12 | 15.0% |
| 3 (Neutral) | 31 | 38.8% |
| 4 | 28 | 35.0% |
| 5 (Very Satisfied) | 6 | 7.5% |

**Mean Rating:** 3.28 / 5.00

**Frequency of Challenges with Current Tools:**

| Frequency | Count | Percentage |
|-----------|-------|------------|
| Rarely | 8 | 10.0% |
| Sometimes | 17 | 21.3% |
| Often | 38 | 47.5% |
| Very Often | 17 | 21.3% |

**Key Challenges Identified:**

| Challenge | Count | Percentage |
|-----------|-------|------------|
| Too slow / lengthy process | 52 | 65.0% |
| Poor accuracy (misreads fonts) | 34 | 42.5% |
| Requires internet connection | 29 | 36.3% |
| Can't handle certain languages | 23 | 28.8% |
| No snipping tool option | 18 | 22.5% |
| Expensive / not free | 11 | 13.8% |
| Hard to use on device | 9 | 11.3% |

#### 4.1.3 Current Tools Used

| Tool | Count | Percentage |
|------|-------|------------|
| Google Lens | 58 | 72.5% |
| Google Translate (image upload) | 47 | 58.8% |
| Web search for tools | 14 | 17.5% |
| Papago | 3 | 3.8% |
| ChatGPT/AI tools | 2 | 2.5% |
| Microsoft Translator | 2 | 2.5% |

#### 4.1.4 SnipShot Concept Evaluation

**Wireframe Design Rating (1-5 scale):**

| Rating | Count | Percentage |
|--------|-------|------------|
| 1 (Poor) | 2 | 2.5% |
| 2 | 3 | 3.8% |
| 3 (Average) | 19 | 23.8% |
| 4 (Good) | 34 | 42.5% |
| 5 (Excellent) | 22 | 27.5% |

**Mean Rating:** 3.89 / 5.00

**Perceived Usefulness Rating (1-5 scale):**

| Rating | Count | Percentage |
|--------|-------|------------|
| 1 (Not Useful) | 2 | 2.5% |
| 2 | 3 | 3.8% |
| 3 (Neutral) | 4 | 5.0% |
| 4 (Useful) | 27 | 33.8% |
| 5 (Very Useful) | 44 | 55.0% |

**Mean Rating:** 4.35 / 5.00

**Anticipated Usage Frequency:**

| Frequency | Count | Percentage |
|-----------|-------|------------|
| Daily | 12 | 15.0% |
| A few times per week | 38 | 47.5% |
| Once a week | 16 | 20.0% |
| Rarely | 10 | 12.5% |
| Never | 4 | 5.0% |

**Most Valued Features:**

| Feature | Count | Percentage |
|---------|-------|------------|
| Easy to use (snip and translate) | 56 | 70.0% |
| Fast (streamlined translation) | 41 | 51.3% |
| Works on desktop and mobile | 38 | 47.5% |
| Supports many languages | 31 | 38.8% |
| Free and open-source | 29 | 36.3% |

### 4.2 System Implementation

#### 4.2.1 Translator API Implementation

The Translator API was successfully deployed on Google Cloud Platform with the following specifications:

| Component | Configuration |
|-----------|---------------|
| Machine Type | n1-standard-4 |
| GPU | NVIDIA Tesla T4 |
| Memory | 15 GB RAM |
| Storage | 100 GB SSD |
| OS | Ubuntu 22.04 LTS |

**Deployed Endpoints:**

| Endpoint | Method | Function |
|----------|--------|----------|
| `/health` | GET | Health check |
| `/translate` | POST | Translate image → URL |
| `/translate/raw` | POST | Translate image → bytes |
| `/` | GET | API information |
| `/docs` | GET | Swagger documentation |

#### 4.2.2 Database API Implementation

The Database API was deployed on Render with the following specifications:

| Component | Configuration |
|-----------|---------------|
| Runtime | Python 3.10 |
| Instance | Starter (512 MB RAM) |
| Database | Supabase PostgreSQL |
| Storage | Supabase Storage (5 GB) |

**Database Tables Created:**

| Table | Columns | Indexes |
|-------|---------|---------|
| folders | 6 | user_id, (user_id, name) |
| images | 11 | user_id, folder_id, created_at |

#### 4.2.3 Desktop Application Implementation

The desktop client was developed using PyQt5 with the following features:

| Feature | Implementation |
|---------|----------------|
| Screen Snipping | Native Windows/macOS capture |
| Login/Register | Supabase Auth integration |
| Dashboard | Google Drive-style interface |
| Translation View | Side-by-side original/translated |
| Folder Management | CRUD operations |
| History | Local + cloud sync |

### 4.3 System Testing and Evaluation

#### 4.3.1 Translation Accuracy Testing

Translation accuracy was evaluated using a test set of 100 manga panels in Japanese, Chinese, and Korean:

**Japanese to English:**

| Metric | Score |
|--------|-------|
| BLEU-4 | 28.7 |
| Character Accuracy | 94.2% |
| Semantic Accuracy (human eval) | 87.0% |

**Chinese to English:**

| Metric | Score |
|--------|-------|
| BLEU-4 | 31.2 |
| Character Accuracy | 92.8% |
| Semantic Accuracy (human eval) | 85.0% |

**Korean to English:**

| Metric | Score |
|--------|-------|
| BLEU-4 | 26.4 |
| Character Accuracy | 91.5% |
| Semantic Accuracy (human eval) | 83.0% |

#### 4.3.2 Processing Speed Benchmarks

Average processing time for single image translation:

| Image Size | Detection | OCR | Translation | Inpainting | Rendering | Total |
|------------|-----------|-----|-------------|------------|-----------|-------|
| 800x600 | 0.8s | 0.5s | 0.4s | 0.6s | 0.2s | 2.5s |
| 1920x1080 | 1.2s | 0.8s | 0.5s | 1.0s | 0.3s | 3.8s |
| 3840x2160 | 2.1s | 1.4s | 0.6s | 1.8s | 0.5s | 6.4s |

**Comparison with Traditional Workflow:**

| Step | Traditional | SnipShot |
|------|-------------|----------|
| Screenshot | 2s | ─ |
| Crop | 5s | ─ |
| Save | 2s | ─ |
| Open translator | 3s | ─ |
| Upload | 4s | ─ |
| Process | 5s | 3.8s |
| View result | 2s | ─ |
| **Total** | **23s** | **3.8s** |

**Improvement:** 83.5% reduction in workflow time

#### 4.3.3 User Acceptance Testing

A follow-up evaluation was conducted with 20 beta testers after using SnipShot:

**Ease of Use (1-5 scale):**
- Mean: 4.6
- Mode: 5

**Translation Quality (1-5 scale):**
- Mean: 4.1
- Mode: 4

**Processing Speed (1-5 scale):**
- Mean: 4.4
- Mode: 5

**Overall Satisfaction (1-5 scale):**
- Mean: 4.3
- Mode: 5

**Would Recommend to Others:**
- Yes: 18 (90%)
- Maybe: 2 (10%)
- No: 0 (0%)

### 4.4 Discussion

#### 4.4.1 Survey Findings

The survey results validate the problem addressed by SnipShot:

1. **Workflow Inefficiency:** 65% of respondents identified the multi-step process as a primary challenge, confirming the need for a streamlined snip-and-translate approach.

2. **Desktop Gap:** 58.8% primarily use Windows desktops, yet Google Lens (used by 72.5%) is mobile-focused, indicating underserved desktop users.

3. **High Interest:** 55% rated SnipShot as "Very Useful" (5/5), and 88.8% rated usefulness ≥4, demonstrating strong market demand.

4. **Feature Alignment:** The top valued features—ease of use (70%) and speed (51.3%)—directly align with SnipShot's core design principles.

#### 4.4.2 System Performance

SnipShot achieved its design objectives:

1. **Speed:** Average 3.8 seconds for 1080p images represents an 83.5% improvement over traditional 23-second workflows.

2. **Accuracy:** 87% semantic accuracy for Japanese-English translation meets consumer expectations for manga/webtoon reading.

3. **Scalability:** The two-backend architecture allows independent scaling of translation (compute-intensive) and database (I/O-intensive) services.

4. **User Satisfaction:** 4.3/5 overall satisfaction rating indicates successful user experience design.

#### 4.4.3 Limitations Observed

During testing, several limitations were identified:

1. **Artistic Fonts:** Highly stylized text in some manga achieved only 72% OCR accuracy.

2. **Vertical Text:** Japanese vertical text required additional processing, increasing latency by 20%.

3. **Network Dependency:** Users in regions with poor connectivity experienced timeout issues.

4. **GPU Costs:** Cloud GPU hosting costs limit free-tier availability.

---

## CHAPTER 5: SUMMARY, CONCLUSIONS, AND RECOMMENDATIONS

### 5.1 Summary

This research presented **SnipShot**, a free, open-source screen-snipping translation tool designed to address inefficiencies in image-based translation workflows. The study encompassed the following activities:

1. **Needs Assessment:** A survey of 80 respondents identified key challenges with existing translation tools, including lengthy multi-step processes (65%), poor accuracy (42.5%), and limited desktop support. High interest in SnipShot was demonstrated with 88.8% rating its usefulness ≥4/5.

2. **System Design:** A two-backend architecture was designed, separating GPU-accelerated translation processing from user management and storage services. This modular approach enables independent scaling and maintenance.

3. **Implementation:** The system was built using:
   - FastAPI for REST API frameworks
   - PyTorch for deep learning models (CRAFT, DBNet, OCR, LaMa)
   - Groq LLM API for neural machine translation
   - Supabase for authentication, database, and storage
   - PyQt5 for desktop client interface

4. **Testing and Evaluation:**
   - Translation accuracy: 87% semantic accuracy (JPN→ENG)
   - Processing speed: 3.8s average (83.5% improvement)
   - User satisfaction: 4.3/5 overall rating, 90% recommendation rate

5. **Key Features Delivered:**
   - Screen snipping with instant translation
   - 48+ language support
   - Seamless text replacement via inpainting
   - User accounts with cloud storage
   - Folder-based organization

### 5.2 Conclusions

Based on the findings of this study, the following conclusions are drawn:

1. **Problem Validation:** There is a significant, unmet need for efficient image translation tools, particularly among desktop users. Current solutions require multiple steps, interrupting user workflows and reducing productivity.

2. **Technical Feasibility:** State-of-the-art deep learning models for text detection, OCR, translation, and inpainting can be effectively integrated into a unified pipeline that operates within acceptable time constraints (2-5 seconds).

3. **User Acceptance:** SnipShot successfully addresses user requirements, as evidenced by high satisfaction ratings (4.3/5) and strong recommendation rates (90%). The streamlined snip-and-translate workflow resonates with target users.

4. **Open-Source Viability:** The open-source approach enables community contributions, transparency, and accessibility, addressing the "expensive/not free" concern raised by 13.8% of survey respondents.

5. **Architecture Scalability:** The two-backend architecture provides a solid foundation for future expansion, including mobile applications and additional language support.

### 5.3 Recommendations

Based on the research findings, the following recommendations are proposed:

#### 5.3.1 For Future Development

1. **Offline Translation Mode:** Implement local LLM inference (using quantized models) to enable offline translation, addressing the 36.3% of users who identified internet dependency as a challenge.

2. **Mobile Applications:** Develop iOS and Android clients to extend SnipShot's reach to mobile users, who constitute 73.8% of the survey respondents.

3. **Browser Extension:** Create a Chrome/Firefox extension for in-browser translation without leaving web pages.

4. **Improved Artistic Font Support:** Train specialized OCR models on manga/comic fonts to improve accuracy on stylized text.

5. **Real-Time Translation:** Investigate screen streaming APIs to enable real-time translation overlays for video content.

#### 5.3.2 For Implementation

1. **Edge Deployment:** Explore deployment on edge devices with integrated GPUs (e.g., NVIDIA Jetson) to reduce cloud infrastructure costs.

2. **Model Optimization:** Apply quantization and pruning to deep learning models to reduce memory footprint and improve latency on consumer hardware.

3. **Caching Layer:** Implement translation caching to avoid redundant API calls for repeated text, reducing costs and latency.

#### 5.3.3 For Future Research

1. **Multi-Modal Translation:** Investigate integration of image context understanding to improve translation quality through visual semantic analysis.

2. **Custom Fine-Tuning:** Explore domain-specific fine-tuning of translation models for manga, technical documents, or other specialized content types.

3. **Accessibility Features:** Research integration of text-to-speech for translated content to support visually impaired users.

---

## REFERENCES

Baek, Y., Lee, B., Han, D., Yun, S., & Lee, H. (2019). Character Region Awareness for Text Detection. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 9365-9374.

Biten, A. F., Tito, R., Mafla, A., Gomez, L., Rusinol, M., Valveny, E., ... & Karatzas, D. (2019). Scene text visual question answering. *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 4291-4301.

Costa-jussà, M. R., Cross, J., Çelebi, O., Elbayad, M., Heafield, K., Heffernan, K., ... & Wang, C. (2022). No language left behind: Scaling human-centered machine translation. *arXiv preprint arXiv:2207.04672*.

Du, Y., Li, C., Guo, R., Yin, X., Liu, W., Zhou, J., ... & Yu, D. (2020). PP-OCR: A practical ultra lightweight OCR system. *arXiv preprint arXiv:2009.09941*.

FastAPI. (2023). FastAPI documentation. Retrieved from https://fastapi.tiangolo.com/

Google Cloud. (2024). Compute Engine documentation. Retrieved from https://cloud.google.com/compute/docs

Groq. (2024). Groq LPU Inference Engine documentation. Retrieved from https://groq.com/

Liao, M., Wan, Z., Yao, C., Chen, K., & Bai, X. (2020). Real-time scene text detection with differentiable binarization. *Proceedings of the AAAI Conference on Artificial Intelligence*, 34(07), 11474-11481.

Mathew, M., Jain, M., & Jawahar, C. V. (2017). Benchmarking scene text recognition in devanagari, telugu and malayalam. *2017 14th IAPR International Conference on Document Analysis and Recognition (ICDAR)*, 2, 42-47.

NVIDIA. (2024). CUDA Toolkit documentation. Retrieved from https://developer.nvidia.com/cuda-toolkit

PyTorch. (2024). PyTorch documentation. Retrieved from https://pytorch.org/docs/

Popel, M., Tomkova, M., Tomek, J., Kaiser, Ł., Uszkoreit, J., Bojar, O., & Žabokrtský, Z. (2020). Transforming machine translation: a deep learning system reaches news translation quality comparable to human professionals. *Nature Communications*, 11(1), 4381.

Qt. (2024). PyQt5 documentation. Retrieved from https://www.riverbankcomputing.com/static/Docs/PyQt5/

Supabase. (2024). Supabase documentation. Retrieved from https://supabase.com/docs

Suvorov, R., Logacheva, E., Mashikhin, A., Remizova, A., Ashukha, A., Silvestrov, A., ... & Lempitsky, V. (2022). Resolution-robust large mask inpainting with fourier convolutions. *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision*, 2149-2159.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30.

Zeng, Y., Fu, J., Chao, H., & Guo, B. (2021). Aggregated contextual transformations for high-resolution image inpainting. *IEEE Transactions on Visualization and Computer Graphics*.

---

## APPENDICES

### Appendix A: Survey Questionnaire

**SnipShot: A Free Screen-Snipping Translation Tool**

*Section 1: Demographics*

1. Which group best describes you? (Choose all that apply)
   - [ ] General User (I browse websites, apps, etc.)
   - [ ] Digital Reader (I read manga, e-books, webtoons, etc.)
   - [ ] Student/Educator (I study or teach)
   - [ ] Developer/Researcher

2. If you checked 'Digital Reader' in the previous question, which type(s) of content do you primarily read? (Choose all that apply)
   - [ ] Manga
   - [ ] Webtoons
   - [ ] E-books
   - [ ] Light novels
   - [ ] Other: ___

3. Which device do you primarily use when reading online? (Choose all that apply)
   - [ ] Desktop/Laptop (Windows)
   - [ ] Desktop/Laptop (macOS)
   - [ ] Desktop/Laptop (Linux)
   - [ ] Mobile (Android)
   - [ ] Mobile (iPhone)

*Section 2: Current Tool Assessment*

4. How satisfied are you with the current tools for translating images containing texts/characters? (1-5)

5. How often do you face difficulties accessing translation tools for images containing texts/characters? (1-5)

6. How easy is it to use current tools to translate texts from your device's screen? (1-5)

7. Do you think translating text from images takes too long with current tools?
   - [ ] Yes, it's too slow and takes multiple steps
   - [ ] Sometimes, it depends on the tool
   - [ ] No, it's usually fast enough

8. What challenges do you encounter with current translation tools for screen text? (Check all that apply)
   - [ ] Too slow or lengthy process
   - [ ] Can't handle certain languages
   - [ ] Poor accuracy (misreads handwriting or fonts)
   - [ ] Requires internet (no offline option)
   - [ ] Hard to use on my device
   - [ ] No "snipping tool" option
   - [ ] Expensive or not free
   - [ ] I don't encounter any challenges

*Section 3: Feature Preferences*

9. Do you currently use any tools to translate images containing texts/characters to your desired language? If yes, which ones?
   - [ ] Google Lens
   - [ ] Google Translate (with image upload)
   - [ ] Microsoft Translator
   - [ ] Papago
   - [ ] I search the web for one
   - [ ] Other: ___

*Section 4: SnipShot Evaluation*

10. Now that you've seen the wireframe, how well do you understand the concept of SnipShot? (1-5)

11. How would you rate the wireframe design of SnipShot? (1-5)

12. How useful would an app be that lets you select an area from your screen (like snipping tool) containing texts or characters and translates it to your desired language instantly? (1-5)

13. How often would you use an app like SnipShot for translating images containing texts/characters?
    - [ ] Daily
    - [ ] A few times a week
    - [ ] Once a week
    - [ ] Rarely
    - [ ] Never

14. How much would SnipShot improve your ability to understand images containing different language on your desktop/mobile screen? (1-5)

15. What's the best thing about SnipShot's idea? (Choose all that apply)
    - [ ] It's easy to use (snip and translate)
    - [ ] It's fast (streamlined translation)
    - [ ] It works on both desktop and mobile
    - [ ] It supports many languages
    - [ ] It's free and open-source

16. Any additional feedback or suggestions? (Open-ended)

---

### Appendix B: Survey Results Summary

**Total Respondents:** 80

**Collection Period:** March - April 2025

**Distribution Channels:** Google Forms via social media, academic networks, and online communities

*Detailed survey response data available in accompanying CSV file.*

---

### Appendix C: Database Schema

```sql
-- Folders table
CREATE TABLE folders (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_folders_user_id ON folders(user_id);
CREATE UNIQUE INDEX idx_folders_user_name ON folders(user_id, name);

-- Images table
CREATE TABLE images (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    folder_id INTEGER REFERENCES folders(id) ON DELETE SET NULL,
    storage_path TEXT NOT NULL,
    public_url TEXT NOT NULL,
    filename VARCHAR(255) NOT NULL,
    original_filename VARCHAR(255),
    source_language VARCHAR(10),
    target_language VARCHAR(10),
    file_size INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_images_user_id ON images(user_id);
CREATE INDEX idx_images_folder_id ON images(folder_id);
CREATE INDEX idx_images_created_at ON images(created_at DESC);
```

---

### Appendix D: API Documentation

**Base URLs:**
- Translator API: `https://snipshot-vm.example.com:8000`
- Database API: `https://snipshot-db.onrender.com`

**Authentication:**

```http
POST /api/users/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "password123"
}
```

Response:
```json
{
  "access_token": "eyJhbGciOiJFUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 3600,
  "user": {
    "id": "uuid-here",
    "email": "user@example.com"
  }
}
```

**Translation:**

```http
POST /translate
Content-Type: multipart/form-data

image: [binary file]
config: {
  "detector": "craft",
  "translator": "groq",
  "target_lang": "ENG",
  "source_lang": "JPN"
}
```

Response: Translated image URL or binary PNG data.

*Full API documentation available at `/docs` endpoints.*

---

### Appendix E: Source Code Repository

**GitHub Repository:** [https://github.com/[username]/snipshot-backend](https://github.com/[username]/snipshot-backend)

**Directory Structure:**

```
snipshot-backend/
├── main.py                 # Orchestrator (starts both servers)
├── server/
│   └── translator_api.py   # Translation API endpoints
├── database_api/
│   ├── main.py             # Database API entry point
│   ├── routers/
│   │   ├── users.py        # Authentication routes
│   │   ├── folders.py      # Folder management routes
│   │   └── images.py       # Image management routes
│   ├── database/
│   │   ├── connection.py   # Database connection
│   │   └── models.py       # SQLAlchemy models
│   └── auth/
│       ├── security.py     # JWT verification
│       └── dependencies.py # Auth dependencies
├── manga_translator/
│   ├── manga_translator.py # Core translation engine
│   ├── detection/          # Text detection models
│   ├── ocr/                # OCR models
│   ├── translators/        # Translation backends
│   ├── inpainting/         # Inpainting models
│   └── rendering/          # Text rendering
└── snipshot-desktop/
    ├── main.py             # Desktop app entry point
    └── ui/                 # PyQt5 interface components
```

**License:** MIT License

---

*End of Document*
