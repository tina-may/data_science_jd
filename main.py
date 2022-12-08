from utilities_functions import getMultipleChoice
from skill_tagger2 import file_path_to_skills_process
from text_process import JDPipeline
SENTENCE_TAGGER = JDPipeline()

def create_sentence_tagged_from_raw_txt():
    SENTENCE_TAGGER.parse_sections_from_file_choice()


if __name__ == '__main__':
    create_sentence_tagged_from_raw_txt() #writes tab delim sentence labels to file
    file_path_to_skills_process(SENTENCE_TAGGER.tagged_output_path) #writes skills to list of skills and prints sentence tagged and
