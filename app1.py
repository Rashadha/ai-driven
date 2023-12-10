from flask import Flask, render_template, request
from vertexai.language_models import TextGenerationModel

app = Flask(__name__)

generation_model = TextGenerationModel.from_pretrained("text-bison@001")

# Define the fixed context
context = """ """
contextMathematics = """
1. Mathematics
i. The Concept of Derivatives in Calculus:
   Derivatives are fundamental concepts in calculus that measure the rate of change of a
   function. They are used in various fields, including physics, engineering, economics,
   and data analysis. To understand derivatives, consider a function that represents the
   position of an object over time. The derivative of this function represents the object's
   velocity, or the rate of change of its position. Similarly, the second derivative represents
   the object's acceleration, or the rate of change of its velocity.

ii. Complex Numbers:
   Complex numbers are an extension of real numbers that include the imaginary unit 'i,'
   defined as the square root of -1. They are represented in the form a + bi, where 'a' and
   'b' are real numbers. Operations with complex numbers involve adding, subtracting,
   multiplying, and dividing their real and imaginary components separately. For example,
   to add two complex numbers (a + bi) + (c + di) = (a + c) + (b + d)i.

iii. Limits in Calculus:
   Limits are a crucial foundation in calculus that help us analyze the behavior of functions
   as they approach a certain point. When we say the limit of a function f(x) as x
   approaches a particular value, say 'a,' we are examining the behavior of f(x) as x gets
   arbitrarily close to 'a.' Limits are used to define derivatives and integrals, the core
   concepts of calculus. They play a vital role in understanding continuity, rates of change,
   and the accumulation of quantities in various mathematical and real-world contexts.
"""

contextTensors = """
2. Tensors
i. Tensors, Their Types, and Applications in Different Fields:
   Tensors are mathematical objects that generalize the concept of vectors and matrices.
   They have components that transform in a specific way under changes of coordinate
   systems. Tensors can be of various orders, such as scalars (0th order), vectors (1st
   order), and matrices (2nd order). In physics and engineering, tensors find widespread
   applications in describing physical quantities like stress, strain, and electromagnetic
   fields. In mathematics, tensors are fundamental to differential geometry and the study
   of manifolds.
"""

contextPhysics = """
3. Physics
i. Quantum Mechanics and Their Implications:
   Quantum mechanics is a fundamental theory in physics that describes the behavior of
   matter and energy at the atomic and subatomic level. It introduces concepts like
   wave-particle duality, quantization of energy, and the uncertainty principle. These
   concepts challenge our classical understanding of the world and have significant
   implications for fields like chemistry, materials science, and modern electronics.

ii. Relativity, Particularly Special Relativity and Its Effects on Time and Space:
   Special relativity is a theory developed by Albert Einstein that explains how space and
   time are interrelated and how they are affected by the motion of objects. One of its key
   concepts is time dilation, which states that time appears to pass slower for objects
   moving relative to an observer. Another concept is length contraction, which states that
   objects in motion appear to be shorter in the direction of their motion. These effects
   become significant at speeds approaching the speed of light.
"""

contextBiology = """
4. Biology
i. Photosynthesis and Its Role in the Ecosystem:
   Photosynthesis is a fundamental process in biology that converts light energy into
   chemical energy in the form of glucose, a sugar molecule. It is the primary source of
   energy for most organisms on Earth. The process occurs in chloroplasts, plant structures
   that contain chlorophyll, the pigment that absorbs light energy. During photosynthesis,
   plants use water, carbon dioxide, and light energy to produce glucose and oxygen. This
   process plays a crucial role in maintaining the balance of oxygen and carbon dioxide in
   the atmosphere.

ii. Evolution and Its Mechanisms of Natural Selection and Genetic Drift:
   Evolution is the process by which species change over time. It is driven by natural
   selection, which favors individuals with traits that make them better suited to their
   environment. These traits are inherited by offspring, leading to changes in the
   population's genetic makeup over generations. Genetic drift, another factor in evolution,
   is the random change in allele frequencies in a population due to chance events. These
   processes have shaped the diversity of life on Earth.

iii. Concept of Natural Selection in Evolution:
   Natural selection is a fundamental mechanism in the process of evolution, as proposed by
   Charles Darwin. It operates on the variation present in a population, where individuals with traits
   that enhance their survival and reproduction are more likely to pass those traits to the next
   generation. Over time, this leads to the gradual adaptation of a population to its environment.
   Natural selection acts as a driving force behind the diversity and complexity of life, shaping the
   traits of organisms in response to environmental challenges.

iv. Molecular Basis of Genetics, Particularly the Role of DNA and RNA:
   DNA (deoxyribonucleic acid) and RNA (ribonucleic acid) are essential molecules in the field of
   genetics. DNA carries genetic information in its double-helix structure, with sequences of
   nucleotides forming genes. RNA, on the other hand, plays a crucial role in protein synthesis by
   transcribing and translating the genetic code from DNA. The intricate processes of DNA
   replication, transcription, and translation are central to understanding how genetic information is
   stored, transmitted, and expressed in living organisms, serving as the molecular basis of
   heredity.

v. Structure of DNA and How It Contributes to the Storage and Transmission of Genetic Information:
   DNA, or deoxyribonucleic acid, has a double-helix structure composed of two long strands
    twisted around each other. Each strand consists of a sugar-phosphate backbone and
    nitrogenous bases—adenine (A), thymine (T), cytosine (C), and guanine (G). Adenine pairs with
    thymine, and cytosine pairs with guanine, forming the complementary base pairs. This structure
    is essential for the storage and transmission of genetic information. The sequence of these base
    pairs encodes the instructions for building and maintaining living organisms. During processes
    like DNA replication and transcription, the complementary base pairing ensures accurate
    duplication and transmission of genetic information."""

contextAI = """
5. AI
i. What Machine Learning Is, Types of Machine Learning, and How It Is Applied in Data Science:
   Machine learning is a subset of artificial intelligence that focuses on developing algorithms and
   models that enable computers to learn from data and make predictions or decisions without
   explicit programming. There are three main types of machine learning: supervised learning,
   unsupervised learning, and reinforcement learning. In data science, machine learning is widely
   applied for tasks such as classification, regression, clustering, and pattern recognition.

ii. The Significance of Data Preprocessing in Data Science:
   Data preprocessing is a crucial step in data science that involves cleaning, transforming, and
   organizing raw data to make it suitable for analysis. It ensures that the data is accurate,
   complete, and relevant. Common techniques include handling missing values, removing
   outliers, scaling features, and encoding categorical variables. Proper data preprocessing
   enhances the quality and reliability of results obtained from machine learning models and
   statistical analyses.

iii. Overfitting in Machine Learning:
   Overfitting occurs in machine learning when a model learns the training data too well, capturing
   noise and irrelevant patterns that do not generalize to new, unseen data. This results in poor
   performance on new data. Overfitting is often caused by excessively complex models.
   Techniques to address overfitting include using simpler models, feature selection, and
   regularization methods. Cross-validation is also employed to assess a model's performance on
   different subsets of the data.

iv. The Role of Exploratory Data Analysis (EDA) in Data Science:
   Exploratory Data Analysis (EDA) is a critical phase in the data science process that involves
   visually and statistically exploring data sets to understand their key characteristics and patterns.
   The objectives of EDA include identifying trends, outliers, and relationships within the data.
   Common techniques used in EDA include summary statistics, data visualization (such as
   histograms, scatter plots, and box plots), and correlation analysis. EDA provides valuable
   insights that guide subsequent modeling and analysis decisions.

v. Big Data:
   Big data refers to large and complex datasets that exceed the processing capabilities of
   traditional data management tools. It is characterized by the three Vs: volume, velocity, and
   variety. Volume represents the sheer size of the data, velocity is the speed at which data is
   generated and processed, and variety refers to the diversity of data types. Big data presents
   challenges in terms of storage, processing, and analysis. However, it also offers opportunities
   for gaining valuable insights, making informed decisions, and discovering patterns that may not
   be apparent in smaller datasets.

vi. The Difference Between Supervised and Unsupervised Learning in Machine Learning:
   Supervised learning involves training a model using labeled data, where the algorithm learns to
   map input data to known output. In unsupervised learning, the algorithm explores patterns and
   relationships in unlabeled data without explicit guidance on the output.

vii. The Concept of Cross-Validation in the Context of Machine Learning:
   Cross-validation is a technique used to assess the performance of a machine learning model by
   dividing the dataset into multiple subsets. The model is trained on several subsets and validated
   on the remaining data, allowing for a more robust evaluation of its generalization ability.

viii. What Is Feature Engineering, and Why Is It Important in Machine Learning:
   Feature engineering involves selecting, transforming, or creating new features from the existing
   data to improve a model's performance. It is crucial in machine learning as the quality of
   features directly impacts the model's ability to learn and make accurate predictions.

ix. How Does the Bias-Variance Tradeoff Influence the Performance of a Machine Learning Model:
   The bias-variance tradeoff refers to the balance between underfitting (high bias) and overfitting
   (high variance). A model with high bias may oversimplify the data, while a high-variance model
   may fit the training data too closely. Achieving an optimal tradeoff is essential for a model to
   generalize well to new, unseen data.

x. Common Distance Metrics Used in Clustering Algorithms:
   Distance metrics, such as Euclidean distance and Manhattan distance, measure the
   dissimilarity between data points in clustering algorithms. These metrics help algorithms
   determine the proximity of points and group similar items together.

xi. How Does Regularization Contribute to Preventing Overfitting in Machine Learning Models:
   Regularization techniques, such as L1 and L2 regularization, add penalty terms to the model's
   cost function. This discourages overly complex models by penalizing large coefficients, helping
   to prevent overfitting and improving a model's ability to generalize to new data.

xii. The Role of a Confusion Matrix in Evaluating the Performance of a Classification Model:
   A confusion matrix is a table that summarizes the performance of a classification model by
   comparing predicted and actual class labels. It includes metrics like true positives, true
   negatives, false positives, and false negatives, providing insights into the model's accuracy and
   error types.

xiii. How Does the Term "One-Hot Encoding" Relate to the Preprocessing of Categorical Variables in
   Machine Learning:
   One-hot encoding is a technique used to convert categorical variables into a binary matrix
   format. Each category is represented by a binary column, and the presence or absence of a
   category is indicated by a 1 or 0, respectively. This encoding allows machine learning algorithms
   to work with categorical data effectively.

xiv. What is the purpose of a ROC curve, and how is it used in evaluating the performance of a
   binary classification model?
   - Answer: A Receiver Operating Characteristic (ROC) curve visually represents the tradeoff
   between true positive rate and false positive rate for different threshold values in a binary
   classification model. It helps assess the model's discrimination ability, and the area under the
   ROC curve (AUC) quantifies the overall performance.

xv. How does the concept of feature scaling contribute to the training of machine learning models?
   - Feature scaling involves standardizing or normalizing input features to a consistent scale. This
   is crucial in machine learning, as it ensures that features with different scales do not unduly
   influence the model. Scaling aids in better convergence during training and prevents certain
   features from dominating others in the learning process.

xvi. The process of model evaluation and selection in machine learning. What are the key metrics
   used to assess the performance of a model, and how does the choice of evaluation metrics
   depend on the specific problem and the nature of the data?
   - Model evaluation and selection are critical steps in the machine learning pipeline, as they
   determine the effectiveness and reliability of a predictive model. The process involves assessing
   the model's performance using various metrics and choosing the most suitable model based on
   these evaluations.
   The first step in model evaluation is typically splitting the dataset into training and testing sets.
   The model is trained on the training set, and its performance is evaluated on the testing set to
   simulate how well it will generalize to new, unseen data.

xvii. Several key metrics are used for model evaluation, depending on the nature of the problem:
   - Accuracy: This is a fundamental metric that measures the ratio of correctly predicted instances
   to the total instances. While accuracy is straightforward, it may not be suitable for imbalanced
   datasets where one class dominates.
   Precision and Recall: Precision is the ratio of true positive predictions to the total predicted
   positives, emphasizing the accuracy of positive predictions. Recall, on the other hand, is the
   ratio of true positives to the total actual positives, focusing on the ability to capture all positive
   instances. Precision and recall are particularly important in scenarios where false positives or
   false negatives have different consequences.
   F1 Score: The F1 score is the harmonic mean of precision and recall, providing a balanced
   measure that considers both false positives and false negatives. It is particularly useful when
   there is an uneven class distribution.
   - Area Under the Receiver Operating Characteristic Curve (AUC-ROC): AUC-ROC evaluates the
   performance of binary classification models by measuring the area under the ROC curve. It
   provides a comprehensive view of the tradeoff between true positive rate and false positive rate
   at various thresholds.
   Mean Squared Error (MSE) or Mean Absolute Error (MAE): metrics such as MSE or MAE
   quantify the difference between predicted and actual values. MSE penalizes larger errors more
   heavily than MAE, making it sensitive to outliers.
   The choice of evaluation metrics depends on the specific goals of the project and the
   characteristics of the data. For instance, in a fraud detection problem, precision might be
   prioritized to minimize false positives, even if it results in lower recall. In a medical diagnosis
   scenario, a balance between precision and recall might be essential to avoid both false positives
   and false negatives.
   Moreover, it's crucial to consider the business or domain-specific implications of model
   performance. In some cases, the cost of false positives and false negatives may differ
   significantly, influencing the choice of evaluation metrics. Therefore, a thoughtful analysis of the
   problem context and careful consideration of metric trade-offs are essential aspects of effective
   model evaluation and selection in data science.

    Illustrate the concept of ensemble learning in machine learning, detailing the principles behind
    ensemble methods and their applications. How do techniques like bagging, boosting, and
    stacking contribute to improving model performance, and under what circumstances would one
    ensemble method be preferred over another?
    Ensemble learning is a powerful paradigm in machine learning that involves combining the
    predictions of multiple models to enhance overall performance. The underlying principle is
    rooted in the idea that aggregating diverse models can mitigate individual model weaknesses,
    leading to a more robust and accurate predictive system.
    Bagging (Bootstrap Aggregating): Bagging involves training multiple instances of the same base
    model on different subsets of the training data, obtained through bootstrapping (sampling with
    replacement). The final prediction is often an average or a voting mechanism across these
    individual models. Popular bagging algorithms include Random Forests, which use decision
    trees as base learners. Bagging helps reduce overfitting and increases stability by leveraging
    the diversity introduced through bootstrap sampling.
    Boosting: Boosting focuses on sequentially training multiple weak learners, each attempting to
    correct the errors of its predecessor. Examples of boosting algorithms include AdaBoost,
    Gradient Boosting, and XGBoost. Boosting assigns different weights to instances based on their
    prediction errors, emphasizing difficult-to-classify instances. This iterative process leads to the
    creation of a strong learner capable of capturing complex patterns in the data.
    Stacking (Stacked Generalization): Stacking involves training multiple diverse models and
    combining their predictions through a meta-model. The meta-model learns to weigh the outputs
    of the base models, effectively leveraging their collective strengths. Stacking can be more
    sophisticated than other ensemble methods, as it allows for a hierarchical combination of
    models, potentially incorporating different types of algorithms in various layers.
    The choice between bagging, boosting, or stacking depends on the characteristics of the data
    and the goals of the modeling task:
    Bagging is often preferred when the base model is prone to high variance, such as in the case
    of decision trees. It helps in stabilizing the predictions and reducing overfitting.
    Boosting is effective when the base models are weak learners, and there is a need to improve
    overall predictive accuracy. It pays more attention to misclassified instances, making it suitable
    for handling imbalanced datasets.
    Stacking is advantageous when a diverse set of models with complementary strengths is
    available. Stacking can capture intricate patterns and relationships in the data by combining the
    unique perspectives of different models.
    In practice, the choice of ensemble method depends on factors like the nature of the data,
    computational resources, and the interpretability of the final model. Ensemble learning, by
    harnessing the collective intelligence of multiple models, stands as a versatile approach in
    improving the robustness and performance of machine learning systems across various
    domains and problem types."""


parameters = {
    'temperature': 0.2,
    'max_output_tokens': 1024,
    'top_p': 0.8,
    'top_k': 40
}

contexts = {
    'mathematics': contextMathematics,
    'tensors': contextTensors,
    'physics': contextPhysics,
    'biology': contextBiology,
    'ai': contextAI
}

@app.route('/get_response', methods=['POST'])
def get_response():
    question = request.form.get('question')
    question_category = request.form.get('question_category')

    # Use question and question_category in your logic

    # Example: Get the context based on the question_category
    if question_category == 'mathematics':
        context = contextMathematics
    elif question_category == 'tensors':
        context = contextTensors
    elif question_category == 'physics':
        context = contextPhysics
    elif question_category == 'biology':
        context = contextBiology
    elif question_category == 'ai':
        context = contextAI
    else:
        context = context  # Use a default context if category is not recognized

    prompt = f"""Answer the question given in the context below:
    Context: {context}?\n
    Question: {question}\n
    Answer:
    """

    response = generation_model.predict(prompt).text
    return render_template('chatbot.html', context=context, question=question, response=response)

if __name__ == '__main__':
    app.run(debug=True)
