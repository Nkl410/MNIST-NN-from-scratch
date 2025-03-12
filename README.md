# MNIST-NN-from-scratch

Dans ce notebook, j’ai implémenté un réseau de neurones simple à deux couches et l’ai entraîné sur le dataset MNIST de reconnaissance de chiffres. L’objectif était de construire ce modèle sans utiliser TensorFlow ni Keras, afin de mieux comprendre la structure et le fonctionnement interne de ce type de réseau.

En codant chaque étape manuellement (initialisation des poids, propagation avant, rétropropagation et mise à jour des paramètres), j’ai pu approfondir ma compréhension des concepts fondamentaux des réseaux de neurones, notamment les dérivées partielles, la descente de gradient et la fonction de coût. Ce projet pédagogique m’a permis d’explorer en détail le calcul matriciel et l’optimisation, des aspects souvent masqués par les frameworks de haut niveau.

Le jeu de données MNIST est diponible ici : https://www.kaggle.com/datasets/hojjatk/mnist-dataset

les formules mathématique utilisé : 

Our NN will have a simple two-layer architecture. Input layer  𝑎[0]  will have 784 units corresponding to the 784 pixels in each 28x28 input image. A hidden layer  𝑎[1]  will have 10 units with ReLU activation, and finally our output layer 𝑎[2]  will have 10 units corresponding to the ten digit classes with softmax activation.

Forward propagation

𝑍[1]=𝑊[1]𝑋+𝑏[1]
𝐴[1]=𝑔ReLU(𝑍[1]))
𝑍[2]=𝑊[2]𝐴[1]+𝑏[2]
𝐴[2]=𝑔softmax(𝑍[2])
Backward propagation

𝑑𝑍[2]=𝐴[2]−𝑌
𝑑𝑊[2]=1𝑚𝑑𝑍[2]𝐴[1]𝑇
𝑑𝐵[2]=1𝑚Σ𝑑𝑍[2]
𝑑𝑍[1]=𝑊[2]𝑇𝑑𝑍[2].∗𝑔[1]′(𝑧[1])
𝑑𝑊[1]=1𝑚𝑑𝑍[1]𝐴[0]𝑇
𝑑𝐵[1]=1𝑚Σ𝑑𝑍[1]
Parameter updates

𝑊[2]:=𝑊[2]−𝛼𝑑𝑊[2]
𝑏[2]:=𝑏[2]−𝛼𝑑𝑏[2]
𝑊[1]:=𝑊[1]−𝛼𝑑𝑊[1]
𝑏[1]:=𝑏[1]−𝛼𝑑𝑏[1]
Vars and shapes

Forward prop

𝐴[0]=𝑋 : 784 x m
𝑍[1]∼𝐴[1] : 10 x m
𝑊[1] : 10 x 784 (as  𝑊[1]𝐴[0]∼𝑍[1] )
𝐵[1] : 10 x 1
𝑍[2]∼𝐴[2] : 10 x m
𝑊[1] : 10 x 10 (as  𝑊[2]𝐴[1]∼𝑍[2] )
𝐵[2] : 10 x 1
Backprop

𝑑𝑍[2] : 10 x m (  𝐴[2] )
𝑑𝑊[2] : 10 x 10
𝑑𝐵[2] : 10 x 1
𝑑𝑍[1] : 10 x m (  𝐴[1] )
𝑑𝑊[1] : 10 x 10
𝑑𝐵[1] : 10 x 1
