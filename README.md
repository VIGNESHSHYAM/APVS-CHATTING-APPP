Below is a comprehensive, well-structured study note for **Unit 3: Semantic and Discourse Analysis** in Natural Language Processing (NLP), tailored for your exam tomorrow. The notes are organized chapter-wise with subchapters, key terms, examples, placeholders for diagrams (since I can’t draw them here, I’ll describe them), case study questions, and conclusions. This is designed to be massive yet easy to understand, ensuring you have all the critical information at your fingertips. Let’s dive in!

---

# Unit 3: Semantic and Discourse Analysis

This unit explores how meaning is represented and analyzed in NLP, from individual words to entire texts. It covers semantic concepts like word meanings and their relationships, as well as discourse-level analysis, including text structure and reference tracking. These concepts are vital for building NLP systems that understand and generate human-like language.

---

## Chapter 1: Representing Meaning

### Overview
Representing meaning is about capturing the semantics of words, sentences, or texts in a way machines can process.

### Subchapter 1.1: Key Concepts
- **Definition**: Converting linguistic meaning into a structured, machine-readable format.
- **Key Terms**:
  - **Lexical Semantics**: Meaning of individual words and their relationships.
  - **Compositional Semantics**: Sentence meaning derived from word meanings and syntax.
  - **Word Representations**: Techniques to encode words (e.g., one-hot encoding, embeddings).
  - **Semantic Roles & Frames**: Assigning roles like agent, action, object (e.g., "John [agent] kicked [action] the ball [object]").
  - **Logical Form**: Representing meaning with formal logic (e.g., predicate logic).
  - **Knowledge Graphs**: Graph structures linking entities and relationships (e.g., "John" → "owns" → "car").
  - **Distributional Semantics**: Words in similar contexts have similar meanings (e.g., "cat" and "dog" appear near "pet").
  - **Sentence & Document-Level Representations**: Encoding larger text units (e.g., TF-IDF, BERT).
  - **Pragmatics & Contextual Meaning**: Meaning influenced by context and intent (e.g., "Can you pass the salt?" as a request).

### Examples
- **One-hot Encoding**: "Cat" = [1, 0, 0], "Dog" = [0, 1, 0] (binary vectors).
- **Word Embeddings**: "King" = [0.7, 0.2, 0.5], "Queen" = [0.8, 0.3, 0.6] (similar vectors for similar meanings).

### Diagram
- **Description**: Draw a side-by-side comparison:
  - Left: One-hot encoding with "cat" and "dog" as sparse vectors.
  - Right: 2D plot of word embeddings with "king," "queen," "man," "woman" clustered based on similarity.

### Case Study Question
- **Question**: How does representing "bank" as an embedding versus one-hot encoding improve its use in a sentence like "He went to the bank to fish"?
- **Answer Hint**: Embeddings capture context (riverbank vs. financial institution), unlike one-hot’s isolation.

### Conclusion
Representing meaning bridges human language and machine understanding, enabling tasks like translation and question-answering.

---

## Chapter 2: Lexical Semantics

### Overview
Lexical semantics studies word meanings and their relationships, foundational for understanding ambiguity.

### Subchapter 2.1: Key Concepts
- **Definition**: Analyzing how words represent concepts and relate to each other.
- **Key Terms**:
  - **Synonymy**: Similar meanings (e.g., "happy" = "joyful").
  - **Antonymy**: Opposite meanings (e.g., "hot" ≠ "cold").
  - **Hyponymy**: Specific-to-general hierarchy (e.g., "dog" → "animal").
  - **Hypernymy**: General-to-specific hierarchy (e.g., "animal" → "dog").
  - **Homonymy**: Same form, different meanings (e.g., "bank" = financial vs. riverbank).
  - **Polysemy**: Multiple related meanings (e.g., "book" = object vs. to reserve).
  - **Meronymy**: Part of a whole (e.g., "wheel" → "car").
  - **Holonymy**: Whole containing parts (e.g., "car" → "wheel").
  - **Collocations**: Words commonly paired (e.g., "strong tea," not "powerful tea").

### Examples
- **Homonymy**: "Bat" in "A bat flew" (animal) vs. "He swung the bat" (sports equipment).
- **Hyponymy**: "Labrador" → "dog" → "animal".

### Diagram
- **Description**: Draw a tree:
  - Top: "Animal" (hypernym).
  - Middle: "Dog" (hyponym of animal, hypernym of Labrador).
  - Bottom: "Labrador" (hyponym).

### Case Study Question
- **Question**: How does lexical semantics distinguish "head" in "head of the company" vs. "head of the table"?
- **Answer Hint**: Polysemy links related meanings (leader vs. top part), aiding disambiguation.

### Conclusion
Lexical semantics provides the framework to decode word meanings, critical for NLP applications like disambiguation.

---

## Chapter 3: Word Senses

### Overview
Word senses refer to the distinct meanings a word can have, depending on context.

### Subchapter 3.1: Key Concepts
- **Definition**: Specific meanings of a word in different contexts.
- **Key Terms**:
  - **Monosemous Words**: Single meaning (e.g., "oxygen").
  - **Polysemous Words**: Multiple related meanings (e.g., "book" = object vs. reserve).
  - **Homonymous Words**: Same form, unrelated meanings (e.g., "bat" = animal vs. equipment).

### Examples
- **Polysemy**: "Book" in "I read a book" vs. "I’ll book a ticket."
- **Homonymy**: "Bank" in "money bank" vs. "river bank."

### Diagram
- **Description**: Draw a branching tree for "bank":
  - Root: "Bank".
  - Left branch: "Financial institution".
  - Right branch: "Riverbank".

### Case Study Question
- **Question**: Why is identifying word senses important in "He saw a bat in the cave"?
- **Answer Hint**: Determines if "bat" is an animal or equipment based on "cave."

### Conclusion
Word senses highlight the complexity of meaning, necessitating context-aware NLP systems.

---

## Chapter 4: Relation between Senses

### Overview
This explores how different senses of words are interconnected.

### Subchapter 4.1: Key Concepts
- **Definition**: Connections between word meanings via linguistic relationships.
- **Key Terms**: Synonymy, antonymy, homonymy, polysemy, hyponymy, hypernymy, meronymy, holonymy (see Chapter 2).

### Examples
- **Polysemy**: "Head" (body part) relates to "head" (leader).
- **Hyponymy**: "Dog" connects to "animal."

### Diagram
- **Description**: Draw a web:
  - Center: "Head".
  - Lines to: "body part," "leader," "top of table" (polysemy examples).

### Case Study Question
- **Question**: How do sense relations help interpret "The head read a book"?
- **Answer Hint**: Polysemy links "head" (leader) to "read," not body part.

### Conclusion
Sense relations enhance meaning comprehension, aiding tasks like disambiguation.

---

## Chapter 5: Word Sense Disambiguation (WSD)

### Overview
WSD determines the correct meaning of a word in context.

### Subchapter 5.1: Key Concepts
- **Definition**: Identifying a word’s intended sense based on surrounding text.
- **Key Terms**:
  - **Dictionary-Based**: Uses resources like WordNet.
  - **Supervised Learning**: Trains on labeled data.
  - **Unsupervised Learning**: Clusters similar contexts.
  - **Knowledge-Based**: Leverages ontologies.

### Examples
- "Bank" in "He went to the bank to withdraw money" = financial institution.
- "Bank" in "The boat neared the bank" = riverbank.

### Diagram
- **Description**: Flowchart:
  - Start: Ambiguous word "bank".
  - Step 1: Analyze context.
  - Step 2: Match with dictionary or model.
  - End: Correct sense (financial or river).

### Case Study Question
- **Question**: How does WSD resolve "bat" in "He caught a bat in the attic"?
- **Answer Hint**: Context "attic" suggests animal, not equipment.

### Conclusion
WSD resolves ambiguity, improving accuracy in translation and search engines.

---

## Chapter 6: Word Embeddings

### Overview
Word embeddings are vector representations capturing word meanings and relationships.

### Subchapter 6.1: Key Concepts
- **Definition**: Dense vectors where similar words are close in space.
- **Key Terms**:
  - **Word2Vec**: Neural network-based embeddings.
  - **GloVe**: Co-occurrence-based embeddings.
  - **FastText**: Subword-aware embeddings.
  - **BERT**: Contextual embeddings.

### Examples
- "King" = [0.7, 0.2, 0.5], "Queen" = [0.8, 0.3, 0.6] (similar vectors).

### Diagram
- **Description**: 2D plot:
  - Plot "king," "queen," "man," "woman" with arrows showing "king - man + woman ≈ queen."

### Case Study Question
- **Question**: How do embeddings improve "I like to bank fish" interpretation?
- **Answer Hint**: Vectors cluster "bank" with fishing terms, not finance.

### Conclusion
Embeddings enhance semantic understanding, powering advanced NLP tasks.

---

## Chapter 7: Word2Vec

### Overview
Word2Vec generates embeddings using neural networks.

### Subchapter 7.1: Key Concepts
- **Definition**: Maps words to vectors preserving relationships.
- **Key Terms**:
  - **CBOW**: Predicts target from context.
  - **Skip-gram**: Predicts context from target.

### Examples
- "King - man + woman ≈ queen" (analogy learned).

### Diagram
- **Description**: Two diagrams:
  - CBOW: "The," "cat," "on," "the" → "sits".
  - Skip-gram: "sits" → "The," "cat," "on," "the".

### Case Study Question
- **Question**: How does Word2Vec interpret "Paris - France + Italy"?
- **Answer Hint**: Predicts "Rome" via vector arithmetic.

### Conclusion
Word2Vec’s efficiency and analogy capture make it a cornerstone of NLP.

---

## Chapter 8: CBOW (Continuous Bag of Words)

### Overview
CBOW predicts a target word from its context.

### Subchapter 8.1: Key Concepts
- **Definition**: Uses surrounding words to predict the middle word.
- **Key Terms**: Context window, neural network.

### Examples
- "The cat sits on the" → "mat."

### Diagram
- **Description**: Neural network:
  - Input: "The," "cat," "on," "the".
  - Output: "sits".

### Case Study Question
- **Question**: Why is CBOW faster than Skip-gram for "The dog barked loudly"?
- **Answer Hint**: Predicts one word ("barked") from multiple context words.

### Conclusion
CBOW excels with frequent words, offering speed in embedding generation.

---

## Chapter 9: Skip-gram and GloVe

### Overview
Skip-gram and GloVe are alternative embedding methods.

### Subchapter 9.1: Skip-gram
- **Definition**: Predicts context from a target word.
- **Key Terms**: Rare words, computational cost.
- **Example**: "mat" → "The," "cat," "sits," "on."

### Subchapter 9.2: GloVe
- **Definition**: Uses co-occurrence statistics.
- **Key Terms**: Global context, matrix factorization.
- **Example**: Builds vectors from word co-occurrence matrix.

### Diagram
- **Description**: Split diagram:
  - Left: Skip-gram ("mat" → context words).
  - Right: GloVe (matrix with rows/columns as words, cells as co-occurrences).

### Case Study Question
- **Question**: How does GloVe differ from Skip-gram in "The cat sleeps"?
- **Answer Hint**: GloVe uses global stats, Skip-gram local context.

### Conclusion
Skip-gram suits rare words, GloVe leverages global patterns, enhancing embeddings.

---

## Chapter 10: Discourse Segmentation

### Overview
Divides text into meaningful segments.

### Subchapter 10.1: Key Concepts
- **Definition**: Breaking text into coherent units.
- **Key Terms**:
  - **Sentence-Level**: Splits into sentences.
  - **Topic Segmentation**: Identifies topic shifts.
  - **Dialogue Segmentation**: Separates speaker turns.
  - **Methods**: Rule-based, ML, neural networks.

### Examples
- "Climate change is serious. AI is advancing." → Two segments.

### Diagram
- **Description**: Text block with lines dividing topics:
  - Segment 1: Climate sentences.
  - Segment 2: AI sentence.

### Case Study Question
- **Question**: Segment "I ate lunch. Then I coded. AI is cool."
- **Answer Hint**: Three segments: eating, coding, AI opinion.

### Conclusion
Segmentation clarifies text structure, aiding summarization and dialogue systems.

---

## Chapter 11: Text Coherence

### Overview
Ensures logical flow in text.

### Subchapter 11.1: Key Concepts
- **Definition**: Smooth, logical idea connection.
- **Key Terms**:
  - **Local Coherence**: Between consecutive sentences.
  - **Global Coherence**: Overall theme consistency.
  - **Cohesive Devices**: "However," "therefore."
  - **Logical Order**: Chronological, cause-effect.
  - **Lexical Cohesion**: Repeated words/synonyms.

### Examples
- "John loves football. He plays daily." (Local coherence).

### Diagram
- **Description**: Flowchart:
  - Sentence 1 → "therefore" → Sentence 2.

### Case Study Question
- **Question**: Why is "I ate. The sky is blue" incoherent?
- **Answer Hint**: No logical connection between ideas.

### Conclusion
Coherence makes text readable and meaningful, vital for NLP.

---

## Chapter 12: Discourse Structure

### Overview
Organizes text for effective communication.

### Subchapter 12.1: Key Concepts
- **Definition**: Text organization framework.
- **Key Terms**:
  - **RST (Rhetorical Structure Theory)**: Hierarchical relations.
  - **Segmentation-Based**: Topic-based divisions.
  - **Dependency Graph**: Sentence connections.

### Examples
- Coherent: "It rained, so I stayed in."

### Diagram
- **Description**: Tree:
  - Root: Main idea.
  - Branches: Cause ("rain"), effect ("stayed in").

### Case Study Question
- **Question**: Structure "I failed. I studied less."
- **Answer Hint**: Cause-effect relation.

### Conclusion
Discourse structure ensures logical text flow, enhancing comprehension.

---

## Chapter 13: Reference Resolution

### Overview
Identifies what expressions refer to.

### Subchapter 13.1: Key Concepts
- **Definition**: Linking words/phrases to entities.
- **Key Terms**:
  - **Coreference Resolution**: Same entity mentions.
  - **Pronominal Anaphora**: Pronoun-to-noun links.
  - **Named Entity Resolution**: Same entity names.
  - **Bridging Resolution**: Implicit links.

### Examples
- "John left. He was tired." ("He" = John).

### Diagram
- **Description**: Text with arrows:
  - "John" → "He."

### Case Study Question
- **Question**: Resolve "She smiled" in "Alice met Mary. She smiled."
- **Answer Hint**: Context needed; could be Alice or Mary.

### Conclusion
Reference resolution clarifies entity references, improving text understanding.

---

## Chapter 14: Pronominal Anaphora Resolution

### Overview
Resolves pronouns to their nouns.

### Subchapter 14.1: Key Concepts
- **Definition**: Linking pronouns to antecedents.
- **Key Terms**:
  - **Pronoun-Antecedent**: "John" → "he."
  - **Cataphora**: Pronoun before noun ("He arrived, John said").
  - **Zero Anaphora**: Implicit subjects (e.g., Japanese).

### Examples
- "Maria spoke. She was loud." ("She" = Maria).

### Diagram
- **Description**: Sentence with arrow from "she" to "Maria."

### Case Study Question
- **Question**: Resolve "he" in "When he came, John laughed."
- **Answer Hint**: Cataphora; "he" = John.

### Conclusion
Pronominal resolution maintains coherence by tracking pronouns.

---

## Chapter 15: Coreference Resolution

### Overview
Links all mentions of an entity.

### Subchapter 15.1: Key Concepts
- **Definition**: Identifying same-entity expressions.
- **Key Terms**:
  - **Pronominal**: "John" → "he."
  - **Nominal**: "Elon Musk" → "the billionaire."
  - **Demonstrative**: "Product" → "this."
  - **Cataphoric**: "He" before "John."

### Examples
- "Elon Musk founded Tesla. The billionaire innovates." ("billionaire" = Elon).

### Diagram
- **Description**: Text with lines connecting "Elon Musk" and "billionaire."

### Case Study Question
- **Question**: Resolve "this" in "Tesla launched a car. This impressed fans."
- **Answer Hint**: "This" = car.

### Conclusion
Coreference resolution tracks entities, enhancing text clarity.

---

## Final Tips for Your Exam
- **Review Key Terms**: Memorize definitions and relationships.
- **Practice Examples**: Apply concepts to new sentences.
- **Sketch Diagrams**: Visualize embeddings, sense trees, discourse structures.
- **Tackle Case Studies**: Use context to solve ambiguities.

Good luck tomorrow! You’ve got this!
