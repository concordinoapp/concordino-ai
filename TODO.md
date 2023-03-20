    - EAST
    - MMOCR

avec c2v EAST

générer images avec fond blanc et entrainer IA :
164 characters donc 164000 data donc :
-> date[:24_000], designation[:100_000], variety[:5_000], region_1[:5_000], province[:5_000], country[:5_000], winery[:20_000]
-> générer data et déteriorer images
    -> couleur image : ou noir, rouge, gold ou blanc sur noir
    -> https://github.com/mastnk/imagedegrade
    -> filtre sur image pour qu'elle soit plus grosse au millieu, comme sur ue bouteille de vin ****et** en courbe**
-> entrainer modèle actuel avec ces images puis tester avec des images prises de EAST
-> tester 2ème modèle si fonctionne mieux : https://pylessons.com/ctc-text-recognition, l'entrainer et le teste de la même manière
-> comparer les 2
-> entrainer un modèle de localisation de texte sur le même principe que les modèle de localisation et reconnaissance d'image

-----------------------------------------------

mieux générer la data : 




