from pipeline import Template, Datatype

class RIVM_template(Template):

    def __init__(self):
        super(RIVM_template,self).__init__()
        self.delimiter= '\t'
        
        self.header = ['id',
                       'forum',
                       'sub_forum',
                       'topic_base_url',
                       'is_topic',
                       'nr_of_pages',
                       'sub_page',
                       'date',
                       'title',
                       'content',
                       'zdate',
                       self.label_name,
                       'topic_id']

        self.types = [Datatype.skip,
                      Datatype.skip,
                      Datatype.skip,
                      Datatype.skip,
                      Datatype.skip,
                      Datatype.skip,
                      Datatype.skip,
                      Datatype.skip,
                      Datatype.content,
                      Datatype.content,
                      Datatype.skip,
                      Datatype.label,
                      Datatype.skip]

        subforum = {'Gezondheid': 0, 'Kinderen': 1, 'Psyche': 2}
        self.dicts = [subforum]

        self.label = {'f':0, 't':1}

        self.artefacts = ['\\n']

        self.number_of_folds = 10
