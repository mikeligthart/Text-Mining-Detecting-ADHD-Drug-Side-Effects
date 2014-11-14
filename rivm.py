import preprocessor

class RIVM_template(preprocessor.Template):

    def __init__(self):
        super(RIVM_template,self).__init__()
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

        self.types = [preprocessor.Datatype.rem,
                      preprocessor.Datatype.rem,
                      preprocessor.Datatype.dct,
                      preprocessor.Datatype.rem,
                      preprocessor.Datatype.bln,
                      preprocessor.Datatype.rem,
                      preprocessor.Datatype.rem,
                      preprocessor.Datatype.rem,
                      preprocessor.Datatype.sst,
                      preprocessor.Datatype.rem,
                      preprocessor.Datatype.zdt,
                      preprocessor.Datatype.lbl,
                      preprocessor.Datatype.rem]

        self.subforum = {'Gezondheid': 0, 'Kinderen': 1, 'Psyche': 2}

        self.dicts = [self.subforum]

        self.label = {'f':0, 't':1}
