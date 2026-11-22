sys_prompt = (
    "Based on the triplets from a knowledge graph, please answer the given question. "
    "Please keep the answers as simple as possible and return all the possible answers "
    """as a list, each with a prefix "ans:"."""
)
cot_prompt = (
    'Format your above answers by listing each answer on a separate line, starting with the prefix "ans:".'
    # " Put your most confident answer first."
    # "2. For each answer, provide reasoning and a likelihood score based on the given triplets."
    # "3. Do not add any new answers."
    # "3. Remove irrelevant answers according to your knowledge, but do not add any additional answers."
)


sys_prompt_gpt = (
    "Based on the triplets retrieved from a knowledge graph, please select relevant triplets for answering the question."
    ' Please return formatted triplets as a list, each prefixed with "evidence:".'
)
cot_prompt_gpt = cot_prompt


sys_prompt_rm_rank = sys_prompt
cot_prompt_rm_rank = (
    "Format your above answers as follows:\n"
    "First, remove wrong answers according to your knowledge.\n"
    'Second, list each answer on a separate line, starting with the prefix "ans:".\n'
    "Finally, put your most confident answer first."
)


sys_prompt_simp = (
    "Based on the triplets from a knowledge graph, please answer the given question."
)
cot_prompt_simp = (
    'Format your answers by listing each answer on a separate line, starting with the prefix "ans:".'
)




# cot_prompt = (
#     "Format your above answers as follows: "
#     '1. List each answer on a separate line, starting with the prefix "ans:". '
#     "2. Put your most confident answer first. "
#     "3. Remove wrong answers according to your knowledge. "
#     "4. If no answer was found from the given triplets, provide answers based on your knowledge, "
#     'each with a prefix "ans:" as well.'
# )


# cot_prompt = (
#     "Format your above answers as follows: "
#     '1. List each answer on a separate line, starting with the prefix "ans:". '
#     "2. Put your most confident answer first. "
#     "3. Remove irrelevant answers according to your knowledge, but do not add any additional answers."
# )


# cot_prompt = (
#     "Format your above answers as follows: "
#     "First, remove irrelevant and wrong answers according to the triplets and your knowledge. "
#     "Second, put your most confident answer first. "
#     'Third, list each answer on a separate line, starting with the prefix "ans:". '
# )

icl_sys_prompt = (
    "Based on the triplets retrieved from a knowledge graph, please answer the question."
    ' Please return formatted answers as a list, each prefixed with "ans:".'
)

# icl_cot_prompt = (
#     # "If no answer is found, please provide an answer according to your common sense. "
#     # "Please keep the answer as simple as possible."
#     #  'Format your answers by listing each answer on a separate line, starting with the prefix "ans:".'
#     #  " Use the entity names as they appear in the knowledge graph as your answers."
#     #  f" Again, the question is: {prompts['question']}"
#     # 'If there is no sufficient information to answer the question, return "ans: not available".'
#     "Let's think step by step."
#     ' Return the most possible answers by listing each answer on a separate line, starting with the prefix "ans:".'
#         # 'Format your above answers by listing each answer on a separate line, starting with the prefix "ans:".'
#     ' Otherwise, if there is no sufficient information to answer the question, return "ans: not available".'

# #    'If there is no sufficient information to answer the question, return "ans: not available".'
# #    'Otherwise, return the most possible answers, each prefixed with "ans:".'

# )

icl_cot_prompt = (
    # "If no answer is found, please provide an answer according to your common sense. "
    # "Please keep the answer as simple as possible."
    #  'Format your answers by listing each answer on a separate line, starting with the prefix "ans:".'
    #  " Use the entity names as they appear in the knowledge graph as your answers."
    #  f" Again, the question is: {prompts['question']}"
    # 'If there is no sufficient information to answer the question, return "ans: not available".'
    "Let's think step by step."
    ' Return the most possible answers based on the given triplets by listing each answer on a separate line, starting with the prefix "ans:".'
        # 'Format your above answers by listing each answer on a separate line, starting with the prefix "ans:".'
    ' Otherwise, if there is no sufficient information to answer the question, return "ans: not available".'

#    'If there is no sufficient information to answer the question, return "ans: not available".'
#    'Otherwise, return the most possible answers, each prefixed with "ans:".'

)

icl_cot_prompt_post = (
    # "If no answer is found, please provide an answer according to your common sense. "
    # "Please keep the answer as simple as possible."
    #  'Format your answers by listing each answer on a separate line, starting with the prefix "ans:".'
    #  " Use the entity names as they appear in the knowledge graph as your answers."
    #  f" Again, the question is: {prompts['question']}"
    # 'If there is no sufficient information to answer the question, return "ans: not available".'
    'Return the most possible answers based on the given triplets by listing each answer on a separate line, starting with the prefix "ans:".'
        # 'Format your above answers by listing each answer on a separate line, starting with the prefix "ans:".'
    ' Otherwise, if there is no sufficient information to answer the question, return "ans: not available".'
    " Let's think step by step."

#    'If there is no sufficient information to answer the question, return "ans: not available".'
#    'Otherwise, return the most possible answers, each prefixed with "ans:".'

)


icl_user_prompt = """Triplets:
(Lou Seal,sports.mascot.team,San Francisco Giants)
(San Francisco Giants,sports.sports_team.championships,2012 World Series)
(San Francisco Giants,sports.sports_championship_event.champion,2014 World Series)
(San Francisco Giants,time.participant.event,2014 Major League Baseball season)
(San Francisco Giants,time.participant.event,2010 World Series)
(San Francisco Giants,time.participant.event,2010 Major League Baseball season)
(San Francisco Giants,sports.sports_team.championships,2014 World Series)
(San Francisco Giants,sports.sports_team.team_mascot,Crazy Crab)
(San Francisco Giants,sports.sports_team.championships,2010 World Series)
(San Francisco Giants,sports.professional_sports_team.owner_s,Bill Neukom)
(San Francisco Giants,time.participant.event,2012 World Series)
(San Francisco,sports.sports_team_location.teams,San Francisco Giants)
(San Francisco Giants,sports.sports_team.arena_stadium,AT&T Park)
(AT&T Park,location.location.events,2012 World Series)
(m.011zsc4_,organization.leadership.organization,San Francisco Giants)
(San Francisco Giants,sports.sports_team.previously_known_as,New York Giants)
(AT&T Park,location.location.events,2010 World Series)
(Crazy Crab,sports.mascot.team,San Francisco Giants)
(New York Giants,baseball.baseball_team.league,National League)
(San Francisco Giants,sports.sports_team.colors,Black)
(San Francisco Giants,sports.sports_team.previously_known_as,New York Gothams)
(m.0k079qm,base.schemastaging.team_training_ground_relationship.team,San Francisco Giants)
(m.0k079ry,base.schemastaging.team_training_ground_relationship.team,San Francisco Giants)
(2010 World Series,time.event.locations,AT&T Park)
(San Francisco Giants,time.participant.event,2012 Major League Baseball season)
(San Francisco Giants,baseball.baseball_team.league,National League)
(m.0crtd80,sports.sports_league_participation.league,National League West)
(San Francisco Giants,sports.sports_team.location,San Francisco)
(San Francisco Giants,sports.sports_team.sport,Baseball)
(m.05n6dtn,baseball.baseball_team_stats.team,San Francisco Giants)


Question:
What year did the team with mascot named Lou Seal win the World Series?"""

icl_ass_prompt = """To find the year the team with mascot named Lou Seal won the World Series, we need to find the team with mascot named Lou Seal and then find the year they won the World Series.

From the triplets, we can see that Lou Seal is the mascot of the San Francisco Giants.

Now, we need to find the year the San Francisco Giants won the World Series.

From the triplets, we can see that San Francisco Giants won the 2010 World Series and 2012 World Series and 2014 World Series.

So, the team with mascot named Lou Seal (San Francisco Giants) won the World Series in 2010, 2012, and 2014.

Therefore, the formatted answers are:

ans: 2014 (2014 World Series)
ans: 2012 (2012 World Series)
ans: 2010 (2010 World Series)"""

# v1
# select_sys_prompt = (
#     "# Role\n"
#     "You are an expert in knowledge graph reasoning, evidence selection, and logical reasoning. "
#     "Your core expertise lies in comprehending the semantics of knowledge-graph triplets, "
#     "identifying the triplets that are most relevant to the question, and using the selected "
#     "evidence to derive concise answers.\n"
#     "# Task Description\n"
#     "Given a question and a set of retrieved knowledge-graph triplets:\n"
#     "- Conduct a step-by-step analysis within <think> tags.\n"
#     "- During the reasoning process, explicitly select the integer indices of the triplets "
#     "that support the answer and place them inside one <select>...</select> block.\n"
#     "- The <select> block must contain only comma-separated integers.\n"
#     "- Use only the selected evidence and necessary logical links to infer the final answer.\n"
#     "# Reasoning Requirements\n"
#     "- Understand the relation path implied by the question.\n"
#     "- Compare retrieved triplets and reject irrelevant or noisy triplets.\n"
#     "- Put only the selected supporting triplet indices inside <select> tags in the middle "
#     "of the <think> reasoning trace, e.g., <select>2, 4, 7</select>. "
#     "If no triplet is sufficient, use <select></select>.\n"
#     "- Synthesize the selected evidence into a faithful answer; do not add unsupported "
#     "answers.\n"
#     "# Output Format\n"
#     "First output the <think>...</think> reasoning trace containing exactly one "
#     "<select>...</select> block with comma-separated integer indices only. Then list each "
#     "final answer on a separate line, starting "
#     "with the prefix \"ans:\". If there is no sufficient information to answer the question, "
#     "return \"ans: not available\"."
# )

select_sys_prompt = (
    "# Role\n"
    "You are an expert in knowledge graph reasoning, evidence selection, and logical reasoning. "
    "Your core expertise lies in comprehending the semantics of knowledge-graph triplets, "
    "identifying the triplets that are most relevant to the question, and using the selected "
    "evidence to derive concise answers.\n"
    "# Task Description\n"
    "Given a question and a set of retrieved knowledge-graph triplets:\n"
    "- Conduct a step-by-step analysis within '<think>' tags, "
    "adhering to the 'Reasoning Requirements', to derive concise answers.\n"
    "# Reasoning Requirements\n"
    "- Select only triplets that are necessary to derive the final answer.\n"
    "- During the reasoning process, enclose the indices of these selected triplets within "
    "<select> tags (e.g., <select>2, 4, 7</select>).\n"
    "- Synthesize the selected evidence into a faithful answer; do not add unsupported "
    "answers.\n"
    "# Output Format\n"
    "Please return formatted answers as a list, each prefixed with \"ans:\"."
)

# v1
# select_cot_prompt = (
#     "Refine your answer using an explicit select-then-answer process. "
#     "First write a concise <think>...</think> reasoning trace and include exactly one "
#     "<select>...</select> block containing only comma-separated integer indices of the "
#     "triplets that directly support the answer, for example <select>2, 4, 7</select>. "
#     "Then list each final answer on a separate line, starting with the prefix \"ans:\". "
#     "If the selected triplets do not provide sufficient information, return \"ans: not available\"."
# )
select_cot_prompt = icl_cot_prompt

noevi_sys_prompt = (
    "Please answer the question."
    ' Please return formatted answers as a list, each prefixed with "ans:".'
)


noevi_cot_prompt = (
    # "If no answer is found, please provide an answer according to your common sense. "
    # "Please keep the answer as simple as possible."
    #  'Format your answers by listing each answer on a separate line, starting with the prefix "ans:".'
    #  " Use the entity names as they appear in the knowledge graph as your answers."
    #  f" Again, the question is: {prompts['question']}"
    # 'If there is no sufficient information to answer the question, return "ans: not available".'
    "Let's think step by step."
    ' Return the most possible answers by listing each answer on a separate line, starting with the prefix "ans:".'
        # 'Format your above answers by listing each answer on a separate line, starting with the prefix "ans:".'
    ' Otherwise, if there is no sufficient information to answer the question, return "ans: not available".'

#    'If there is no sufficient information to answer the question, return "ans: not available".'
#    'Otherwise, return the most possible answers, each prefixed with "ans:".'

)
