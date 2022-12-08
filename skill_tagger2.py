import spacy
from spacy.tokens import DocBin
import json
import re


def str_ents2brackets(doc_):
    out = ''
    docj = doc_.to_json()
    ents = docj["ents"]
    if ents == []:
        return docj["text"]

    head = ents.pop(0)
    s,e = head["start"], head["end"]

    for i,ch in enumerate(docj["text"]):



        if ents == []:
            out += docj["text"][i:]
            break



        if i == s:
            out += "{"
        if i == e - 1:
            out += ch + "}"
            head = ents.pop(0)
            s, e = head["start"], head["end"]
        else:
            out += ch
    return out



def textfile2jsonfile(textfile,jsonfile):
    def string2offsets(str):
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

    json_template = {"annotations":[], "classes":["SKILL"]}
    with open(textfile) as t:
        for line in t:
            [newst,offsets] = string2offsets(line)
            entities = []
            for [s,e] in offsets:
                entities.append([s,e,"SKILL"])
            json_template["annotations"].append([newst,{"entities":entities}])
    with open(jsonfile,"w") as j:
        j.write(json.dumps(json_template,indent=2))


def sentences2skillsfiles(sentences):
    nlp = spacy.load("output_tagger/model-last")
    skills_list = []


    with open(sentences) as st:
        for line in st:
            if "skills_qualifications" not in line:
                continue
            doc = nlp(line)
            skills = re.findall("{.+?}",str_ents2brackets(doc).strip())
            for skill in skills:
                skills_list.append(skill.strip("{}").lower())
    skills_list.sort()
    file_base = sentences.split("/")[-1]
    file_base = file_base.split(".")[0]
    skill_list_file = "text_data/jd_skills_lists/" + file_base + ".txt"
    with open(skill_list_file, "w") as skill_list_file_w:
        for skill in skills_list:
            skill_list_file_w.write(skill + "\n")


def file_path_to_skills_process(path_to_tagged_sentences):
    nlp = spacy.load("output_tagger/model-last")

    textfile2jsonfile(path_to_tagged_sentences, "tmp.0.json")
    with open(path_to_tagged_sentences) as st:
        for line in st:
            doc = nlp(line)
            print(str_ents2brackets(doc).strip())

    sentences2skillsfiles(path_to_tagged_sentences)


nlp = spacy.blank("en")
docbin = DocBin()
with open("text_data/tmp.0.json") as f:
    TRAINING_DATA = json.load(f)

for text, annot in TRAINING_DATA["annotations"]:
    doc = nlp.make_doc(text)
    ents = []
    for start,end,label in  annot["entities"]:
        span = doc.char_span(start,end,label=label, alignment_mode="contract")
        if not span:
            continue
        ents.append(span)
    doc.ents = ents
    #print(doc.ents)
    docbin.add(doc)

docbin.to_disk("./skills_tagged.train.spacy")




#$ python -m spacy train config_skills.cfg --paths.train ./skills_tagged.train.spacy --paths.dev ./skills_tagged.train.spacy -o ./output_tagger/
