import pipeline

class RIVM_template(pipeline.Template):

    def __init__(self):
        super(RIVM_template,self).__init__()
        self.delimiter= '\t'
        
        self.headers = ['id',
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

        self.types = [pipeline.Datatype.rem,
                      pipeline.Datatype.rem,
                      pipeline.Datatype.rem,
                      pipeline.Datatype.rem,
                      pipeline.Datatype.rem,
                      pipeline.Datatype.rem,
                      pipeline.Datatype.rem,
                      pipeline.Datatype.rem,
                      pipeline.Datatype.rem,
                      pipeline.Datatype.con,
                      pipeline.Datatype.rem,
                      pipeline.Datatype.lbl,
                      pipeline.Datatype.rem]

        subforum = {'Gezondheid': 0, 'Kinderen': 1, 'Psyche': 2}
        self.dicts = [subforum]

        self.label = {'f':0, 't':1}

        self.artefacts = ['\\n']

        self.number_of_folds = 10
