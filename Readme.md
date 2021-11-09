Image and Sentence Matching via Semantic
Concepts and Order Learning
Yan Huang, Qi Wu, Wei Wang, and Liang Wang, Senior Member, IEEE
Abstract—Image and sentence matching has made great progress recently, but it remains challenging due to the existing large
visual-semantic discrepancy. This mainly arises from two aspects: 1) images consist of unstructured content which is not semantically
abstract as the words in the sentences, so they are not directly comparable, and 2) arranging semantic concepts in different semantic
order could lead to quite diverse meanings. The words in the sentences are sequentially arranged in a grammatical manner, while the
semantic concepts in the images are usually unorganized. In this work, we propose a semantic concepts and order learning framework
for image and sentence matching, which can improve the image representation by first predicting semantic concepts and then
organizing them in a correct semantic order. Given an image, we first use a multi-regional multi-label CNN to predict its included
semantic concepts in terms of object, property and action. These word-level semantic concepts are directly comparable with the words
of noun, adjective and verb in the matched sentence. Then, to organize these concepts and make them express similar meanings as
the matched sentence, we use a context-modulated attentional LSTM to learn the semantic order. It regards the predicted semantic
concepts and image global scene as context at each timestep, and selectively attends to concept-related image regions by referring to
the context in a sequential order. To further enhance the semantic order, we perform additional sentence generation on the image
representation, by using the groundtruth order in the matched sentence as supervision. After obtaining the improved image
representation, we learn the sentence representation with a conventional LSTM, and then jointly perform image and sentence
matching and sentence generation for model learning. Extensive experiments demonstrate the effectiveness of our learned semantic
concepts and order, by achieving the state-of-the-art results on two public benchmark datasets.
Index Terms—semantic concept, semantic order, context-modulated attention, image and sentence matching
