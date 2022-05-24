# KTH DD2417-Language Engineering

Question answering (QA) is the task of finding some concise, concrete, and correct answer
to a given question. This problem is well-studied with both deep learning and knowledge
based models having been proposed, and well-performing models have been constructed.
A different but related problem is question word prediction (QWP), the problem of
finding the question word missing from a given question and its corresponding answer. To
illustrate, given the following question-answer pair (QAP):

Question: "[qw] is the capital of Sweden?"
Answer: "Stockholm"
  
the correct output should be “What“. The constraints on the problem domain are in this sense
not obvious: should “Which city“ also be considered a valid question phrase (QP), and so a
member of the set of correct answers for above QAP?
  
This project aims to solve this problem using multi-label text classification, where each labels is considered a question word (QW). An overall test accuracy of 94% was achieved when applying the model to the SQuAD dataset.
