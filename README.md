# Predilex ENS Challenge
###  NLP applied to judicial decisions parsing 

Predilex has “jurisprudence” data as text files. The goal of this [challenge](https://challengedata.ens.fr/participants/challenges/24/) is to build an algorithm to automate the extraction of these relevant information:
- sex of the victim:<br>
	This information is always contained in the document and can only take two values : "homme" and "femme".
- date of the accident:<br>
	Except in very rare cases, this information is always domewhere in the document (usually at the beginning). It is the date when the accident happenned. We expect a date in the format dd/mm/yyyy.
- date of the consolidation of the injuries:<br>
	This is the date when the injuries of the victim became stable and were declared final by a physician. The information should be present in most cases but sometimes it is either missing (so we put "n.c." in the csv file) or not applicable (so we put "n.a." in the csv) if the injury did not stabilize before the death of the victim.

This is a project I've done in collaboration with one of my friends. The proposed solution consists in building an xgboost classifier for each information we want to extract. We focused our work on features engineering using text processing techniques and bag of words.