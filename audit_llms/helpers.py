SYSTEM_PROMPT = ('You are a helpful AI assistant with expertise in identifying populism in public discourse.\n\n'
                 'Populism can be defined as an anti-elite discourse in the name of the "people". '
                 'In other words, populism emphasizes the idea of the common "people" and '
                 'often positions this group in opposition to a perceived elite group.\n\n'
                 'There are two core elements in identifying populism: '
                 '(i) anti-elitism, i.e., negative invocations of "elites", and '
                 '(ii) people-centrism, i.e., positive invocations of the "people".\n\n')

INSTRUCTION = ('You must classify each sentence in one of the following categories:\n\n'
               '(a) No populism.\n'
               '(b) Anti-elitism, i.e., negative invocations of "elites".\n'
               '(c) People-centrism, i.e., positive invocations of the "People".\n'
               '(d) Both people-centrism and anti-elitism populism.\n\n')

ALT_INSTRUCTION = ('You must classify each sentence in one of the following categories:\n\n'
               '(b) Anti-elitism, i.e., negative invocations of "elites".\n'
               '(c) People-centrism, i.e., positive invocations of the "People".\n'
               '(d) Both people-centrism and anti-elitism populism.\n\n'
                   '(a) No populism.\n')

DEMONSTRATION_EXAMPLES = 'The following sentences are in category {}:\n\n'

QUERY = 'Which is the most relevant category for the sentence: "{}"?'

RESPONSE_START = 'I would categorize this sentence as ('
