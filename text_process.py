import json

import spacy
from spacy.tokens import DocBin
from spacy.util import minibatch, compounding
from spacy.training.example import Example
from utilities_functions import getMultipleChoice
#import classy_classification



import os
import random
import re




def get_filename():
    tmp = [file_.replace(".raw.txt", "") for file_ in os.listdir("./text_data/jd_raw/")]
    to_multichoice = [choice for choice in tmp if not re.findall("^\d+$",choice)]
    return getMultipleChoice(to_multichoice)

class JDPipeline:
    def __init__(self):
        nlp_tagger = spacy.load('en_core_web_sm')
        self.nlp = spacy.load('en_core_web_sm')
        self.sentence_tagset = self.get_tsv_cats() #['other', 'company_info', 'misc_job_info', 'skills_qualifications', 'values_disclaimers']


    # break into sentences
    def sentence_breaker(self,text):
        tokens = self.nlp(text)
        return [sent for sent in tokens.sents]

    def sentence_corpus_cleaner(self,corpus):
        def sent_norm(s):
            s =  re.sub("^[ ·*·\t]+","",s.text.strip())
            s = s.replace("\n"," ")
            s = s.replace("\u2019","'")
            s = s.replace("\u2013","-")
            s = re.sub(" +", " ",s)
            return s
        out = []
        for sent in corpus:
            sent = sent_norm(sent)
            if sent != '': out.append(sent)
        return out

    def string2offsets(self,str):
        newst = ''
        offsets = []
        index = 0
        for ch in str:
            if ch not in "{}":
                index += 1
                newst += ch
            elif ch == "{":
                offsets.append([index,-1])
            else:
                offsets[-1][-1] = index
        return [newst,offsets]

    def get_tsv_cats(self):
        labels = set([])
        with open("text_data/sentences_tagged.tsv") as tagged:
            curr_set = "dev" if random.randint(1, 10) == 1 else "train"
            for line in tagged:
                [label, text] = line.strip().split("\t")
                labels.add(label)
        return list(labels)

    def tsv_to_textcat(self):
        annotations = []
        data_json = {"annotations":[], "classes": self.sentence_tagset}

        with open("text_data/sentences_tagged.tsv") as tagged:
            curr_set = "dev" if random.randint(1, 10) == 1 else "train"
            for line in tagged:
                [label, text] = line.strip().split("\t")

                example_json = {label:0.0 for label in self.sentence_tagset}
                example_json[label] = 1.0
                annotation = [text, {"cats":example_json}]
                annotations.append(annotation)
        data_json["annotations"] = annotations

        nlp = spacy.blank("en")
        docbin = DocBin()
        with open("text_data/tmp.cats.json","w") as f:
            f.write(json.dumps(data_json,indent=4))

        for text, annot in data_json["annotations"]:
            doc = nlp.make_doc(text)
            cats = annot["cats"]
            doc.cats = cats
            # print(doc.ents)
            docbin.add(doc)

        docbin.to_disk("./sentences_tagged.train.spacy")

        #cat_nlp = spacy.blank("en")
        #cat_nlp.add_pipe("textcat_multilabel", config={"data":data, "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "device":"cpu"})





    # train sentence classifier
    def sentence_classifier_setup_training(self):

        labels = self.sentence_tagset  # ['other', 'company_info', 'misc_job_info', 'skills_qualifications', 'values_disclaimers']
        db_train = DocBin(attrs=["LEMMA", "POS"])

        db_dev = DocBin(attrs=["LEMMA", "POS"])
        sets_ = {"train": db_train, "dev": db_dev}

        with open("text_data/sentences_tagged.tsv") as tagged:
            curr_set = "dev" if random.randint(1, 10) == 1 else "train"
            for line in tagged:
                [label, text] = line.strip().split("\t")
                doc = pipeline.nlp(text)
                doc.cats = {label_: (label == label_) for label_ in labels}
                sets_[curr_set].add(doc)
            db_train.to_disk("./sentences_tagged.train.spacy")
            db_dev.to_disk("./sentences_tagged.dev.spacy")


    # runtime break into sections
    def runtime_parse_sections(self,path_):
        sent_classifier = spacy.load("output/model-last")
        sections = {}
        all_sents = []

        with open(path_) as o:
            filetxt = o.read()
            for sent in self.sentence_corpus_cleaner(self.sentence_breaker(filetxt)):
                cats = sent_classifier(sent).cats
                label = None
                max_conf = 0
                for label_ in cats:
                    if cats[label_] > max_conf:
                        max_conf = cats[label_]
                        label = label_
                if label in sections:
                    sections[label].append(sent)
                else:
                    sections[label] = [sent]
        outpath = path_.replace(".raw.",".sentences_tagged.")
        outpath = "text_data/jd_tagged_sentences/" + outpath.split("/")[-1]
        self.tagged_output_path = outpath
        with open(outpath,"w") as x:
            for section in sections:
                for sentence in sections[section]:
                    x.write("\t".join([section,sentence + "\n"]))


        #print(json.dumps(sections,indent=4))




    # train parameter extractor
    # apply parameter extraction model
    def parse_sections_from_file_choice(self):



        nlp = spacy.load("output/model-last")
        raw_filename = get_filename()
        self.runtime_parse_sections("./text_data/jd_raw/{0}.raw.txt".format(raw_filename))




p = JDPipeline()
p.tsv_to_textcat()

#python -m spacy train janice_config.cfg --paths.train ./sentences_tagged.train.spacy --paths.dev ./sentences_tagged.train.spacy -o ./output








