# Projet Cuda

## Auteurs

---

- Nabyl QUIGNON
- Mugilan RAGAVAN
- Terence CLASTRES
---

## Build

Chaque filtre a son propre Makefile. Vous pouvez utiliser la commande "make all" pour générer tous les exécutables. Pour exécuter un programme, utilisez l'exécutable correspondant en lui fournissant une image à traiter en tant qu'argument. 
Par exemple, exécutez "./edgeCuda ../images/in.jpg" pour exécuter le filtre "edgeCuda" sur l'image "../images/in.jpg".

De plus, un fichier pratique appelé "test.sh" est disponible pour exécuter un filtre un certain nombre de fois. La syntaxe de la commande est la suivante : ./test.sh ./nomdelexecutable nbrepetition /chemin/vers/image.jpg. Par exemple, vous pouvez exécuter le filtre "edgeCuda" 10 fois sur l'image "../images/in.jpg" avec la commande ./test.sh ./edgeCuda 10 ../images/in.jpg.

En plus d'exécuter le filtre plusieurs fois, le fichier "test.sh" calcule également le temps moyen d'exécution des 10 exécutions. Une fois les exécutions terminées, le temps moyen d'exécution sera affiché, ce qui vous donnera une idée de la performance du filtre sur l'image spécifiée.

## Éxécution sur un GPU qui n'est pas de marque Nvidia

Comme nous avons eu des problèmes pour accéder à la carte nvidia de l'université, nous avons cherché des méthodes alternatives : 
L'une d'entre elles permet de convertir le code cuda vers un format plus générique qui s'éxécute à la fois sur des GPUs Nvidia et AMD sans surcoût important.

Le fichier executable `hipify.sh` à la racine du dossier permet de traduire un fichier .cu en fichier .hip, puis de générer un exéctuable.

C'est HIP (https://docs.amd.com/bundle/HIP-Programming-Guide-v5.3/page/Introduction_to_HIP_Programming_Guide.html), une initative d'AMD, qui permet d'avoir un code commun s'éxécutant sur des GPUs de marque différentes 



