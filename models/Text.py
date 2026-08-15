class_names_5 = [
    "Neutral (student in class).",
    "Enjoyment (student in class).",
    "Confusion (student in class).",
    "Fatigue (student in class).",
    "Distraction (student in class)."
]

class_names_with_context_5 = [
    "A student shows a neutral learning state in a classroom.",
    "A student shows enjoyment while learning in a classroom.",
    "A student shows confusion during learning in a classroom.",
    "A student shows fatigue during learning in a classroom.",
    "A student shows distraction and is not focused in a classroom."
]

class_descriptor_5_only_face = [
    "A student has a neutral face with relaxed mouth, open eyes, and calm eyebrows.",
    "A student looks happy with a slight smile, bright eyes, and relaxed eyebrows.",
    "A student looks confused with furrowed eyebrows, a puzzled look, and slightly open mouth.",
    "A student looks tired with drooping eyelids, frequent yawning, and a sleepy face.",
    "A student looks distracted with unfocused eyes and a wandering gaze away from the lesson."
]

class_descriptor_5_only_body = [
    "A student sits still with an upright posture and hands on the desk, showing a neutral learning state.",
    "A student leans slightly forward with an open, engaged posture, showing enjoyment in learning.",
    "A student tilts the head and leans in, hand on chin, showing confusion while trying to understand.",
    "A student slouches with shoulders dropped and head lowered, showing fatigue during class.",
    "A student shifts around, turns away from the desk, or looks sideways, showing distraction and low focus."
]

class_descriptor_5 = [
    "A student looks neutral and calm in class, with a relaxed face and steady gaze, quietly watching the lecture or reading notes.",
    "A student shows enjoyment while learning, with a gentle smile and bright eyes, appearing engaged and interested in the lesson.",
    "A student looks confused in class, with furrowed eyebrows and a puzzled expression, focusing on the material as if trying to understand.",
    "A student appears fatigued in class, with drooping eyelids and yawning, head slightly lowered, showing low energy.",
    "A student is distracted in class, frequently looking away from the lesson, scanning around, and not paying attention to learning materials."
]

# Prompt Ensemble for RAER (5 classes)
# Each inner list contains multiple descriptions for a single class.
prompt_ensemble_5 = [
    [   # Neutral
        "A photo of a student being alert and looking straight ahead.",
        "A photo of a student with a calm and steady gaze.",
        "A photo of a student paying attention with a neutral expression."
    ],
    [   # Enjoyment
        "A photo of a student smiling and looking happy.",
        "A photo of a student showing joy and enthusiasm.",
        "A photo of a student appearing pleased and engaged."
    ],
    [   # Confusion
        "A photo of a student frowning with a puzzled expression.",
        "A photo of a student scratching their head or looking confused.",
        "A photo of a student trying hard to understand but failing."
    ],
    [   # Fatigue
        "A photo of a student yawning or falling asleep.",
        "A photo of a student with heavy drooping eyelids.",
        "A photo of a student resting their head, looking very tired."
    ],
    [   # Distraction
        "A photo of a student looking away from the screen.",
        "A photo of a student turning their head to the side.",
        "A photo of a student engaging in other activities, not studying."
    ]
]

class_descriptor_5_au = [
    "A student with AU0 neutral face and AU43 eyes normally open, showing neutrality.",
    "A student with AU6 cheek raiser and AU12 lip corner puller, showing enjoyment.",
    "A student with AU4 brow lowerer and AU7 lid tightener, showing confusion.",
    "A student with AU43 eyes closed and AU46 drooping eyelids, showing fatigue.",
    "A student with AU51 head turn left or AU52 head turn right, showing distraction."
]

class_descriptor_8 = [
    'A person who is feeling neutral.',
    'A person who is feeling happy.',
    'A person who is feeling sad.',
    'A person who is feeling surprise.',
    'A person who is feeling fear.',
    'A person who is feeling disgust.',
    'A person who is feeling anger.',
    'A person who is feeling contempt.'
]

class_names_8 = [
    'Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt'
]

class_names_7 = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger']

class_descriptor_7 = [
    'A person who is feeling neutral.',
    'A person who is feeling happy.',
    'A person who is feeling sad.',
    'A person who is feeling surprise.',
    'A person who is feeling fear.',
    'A person who is feeling disgust.',
    'A person who is feeling anger.'
]

class_names_with_context_7 = [
    'A person shows neutral emotion.',
    'A person shows happy emotion.',
    'A person shows sad emotion.',
    'A person shows surprise emotion.',
    'A person shows fear emotion.',
    'A person shows disgust emotion.',
    'A person shows anger emotion.'
]

class_descriptor_7_only_face = [
    'The face of a person who is feeling neutral.',
    'The face of a person who is feeling happy.',
    'The face of a person who is feeling sad.',
    'The face of a person who is feeling surprise.',
    'The face of a person who is feeling fear.',
    'The face of a person who is feeling disgust.',
    'The face of a person who is feeling anger.'
]

class_descriptor_7_only_body = [
    'The body of a person who is feeling neutral.',
    'The body of a person who is feeling happy.',
    'The body of a person who is feeling sad.',
    'The body of a person who is feeling surprise.',
    'The body of a person who is feeling fear.',
    'The body of a person who is feeling disgust.',
    'The body of a person who is feeling anger.'
]

class_names_with_context_8 = [
    'A person shows neutral emotion.',
    'A person shows happy emotion.',
    'A person shows sad emotion.',
    'A person shows surprise emotion.',
    'A person shows fear emotion.',
    'A person shows disgust emotion.',
    'A person shows anger emotion.',
    'A person shows contempt emotion.'
]

class_descriptor_8_only_face = [
    'The face of a person who is feeling neutral.',
    'The face of a person who is feeling happy.',
    'The face of a person who is feeling sad.',
    'The face of a person who is feeling surprise.',
    'The face of a person who is feeling fear.',
    'The face of a person who is feeling disgust.',
    'The face of a person who is feeling anger.',
    'The face of a person who is feeling contempt.'
]

class_descriptor_8_only_body = [
    'The body of a person who is feeling neutral.',
    'The body of a person who is feeling happy.',
    'The body of a person who is feeling sad.',
    'The body of a person who is feeling surprise.',
    'The body of a person who is feeling fear.',
    'The body of a person who is feeling disgust.',
    'The body of a person who is feeling anger.',
    'The body of a person who is feeling contempt.'
]

# CK+ Classes (Alphabetical Order: Anger, Contempt, Disgust, Fear, Happy, Sadness, Surprise)
class_names_ckplus = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise']

class_names_with_context_ckplus = [
    "A person shows anger.",
    "A person shows contempt.",
    "A person shows disgust.",
    "A person shows fear.",
    "A person shows happiness.",
    "A person shows sadness.",
    "A person shows surprise."
]

class_descriptor_ckplus = [
    "A person with an angry expression, furrowed brows and tightened lips.",
    "A person with a contemptuous expression, one corner of the lip raised.",
    "A person with a disgusted expression, nose wrinkled and upper lip raised.",
    "A person with a fearful expression, eyes wide open and eyebrows raised.",
    "A person with a happy expression, smiling with cheeks raised.",
    "A person with a sad expression, corners of the lips turned down and drooping eyelids.",
    "A person with a surprised expression, mouth open and eyes widened."
]

prompt_ensemble_ckplus = [
    [ # Anger
        "A photo of a person showing anger.",
        "A face with furrowed brows and a glare.",
        "An angry facial expression."
    ],
    [ # Contempt
        "A photo of a person showing contempt.",
        "A face with a smirk or sneer.",
        "A contemptuous facial expression."
    ],
    [ # Disgust
        "A photo of a person showing disgust.",
        "A face with a wrinkled nose.",
        "A disgusted facial expression."
    ],
    [ # Fear
        "A photo of a person showing fear.",
        "A face with wide eyes and a terrified look.",
        "A fearful facial expression."
    ],
    [ # Happy
        "A photo of a person showing happiness.",
        "A smiling face with joy.",
        "A happy facial expression."
    ],
    [ # Sadness
        "A photo of a person showing sadness.",
        "A face with a frown and sorrowful eyes.",
        "A sad facial expression."
    ],
    [ # Surprise
        "A photo of a person showing surprise.",
        "A face with an open mouth and wide eyes.",
        "A surprised facial expression."
    ]
]

# SFER Classes (Alphabetical: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise)
class_names_sfer = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

class_names_with_context_sfer = [
    "A student shows anger.",
    "A student shows disgust.",
    "A student shows fear.",
    "A student shows happiness.",
    "A student shows neutrality.",
    "A student shows sadness.",
    "A student shows surprise."
]

class_descriptor_sfer = [
    "A student with an angry expression, furrowed brows and tightened lips.",
    "A student with a disgusted expression, nose wrinkled and upper lip raised.",
    "A student with a fearful expression, eyes wide open and eyebrows raised.",
    "A student with a happy expression, smiling with cheeks raised.",
    "A student with a neutral expression, relaxed face and calm gaze.",
    "A student with a sad expression, corners of the lips turned down and drooping eyelids.",
    "A student with a surprised expression, mouth open and eyes widened."
]

prompt_ensemble_sfer = [
    [ # Anger
        "A photo of a student showing anger.",
        "A face with furrowed brows and a glare.",
        "An angry facial expression."
    ],
    [ # Disgust
        "A photo of a student showing disgust.",
        "A face with a wrinkled nose.",
        "A disgusted facial expression."
    ],
    [ # Fear
        "A photo of a student showing fear.",
        "A face with wide eyes and a terrified look.",
        "A fearful facial expression."
    ],
    [ # Happy
        "A photo of a student showing happiness.",
        "A smiling face with joy.",
        "A happy facial expression."
    ],
    [ # Neutral
        "A photo of a student showing a neutral expression.",
        "A calm face with no strong emotion.",
        "A neutral facial expression."
    ],
    [ # Sad
        "A photo of a student showing sadness.",
        "A face with a frown and sorrowful eyes.",
        "A sad facial expression."
    ],
    [ # Surprise
        "A photo of a student showing surprise.",
        "A face with an open mouth and wide eyes.",
        "A surprised facial expression."
    ]
]

# CAER Classes (Alphabetical: Anger, Disgust, Fear, Happy, Neutral, Sad, Surprise)
class_names_caer = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

class_names_with_context_caer = [
    "A person shows anger.",
    "A person shows disgust.",
    "A person shows fear.",
    "A person shows happiness.",
    "A person shows neutrality.",
    "A person shows sadness.",
    "A person shows surprise."
]

class_descriptor_caer = [
    "A person displaying intense anger, with deeply furrowed brows, a hardened glare, flared nostrils, and tightly compressed lips or an open mouth shouting, often in a tense or confrontational scene.",
    "A person showing strong disgust, characterized by a severely wrinkled nose, raised upper lip, squinted eyes, and a repulsed posture, reacting to an unpleasant context.",
    "A person exhibiting genuine fear, with wide-open eyes showing the sclera, raised and drawn-together eyebrows, and a tense, slightly open mouth, often in an alarming or threatening scene.",
    "A person experiencing pure happiness, featuring a bright, genuine smile (Duchenne smile) with raised cheeks, crinkled eyes, and relaxed posture, often in a joyful or celebratory context.",
    "A person with a completely neutral demeanor, showing relaxed facial muscles, a calm and steady gaze, and no distinct emotional micro-expressions, set in an ordinary everyday scene.",
    "A person expressing deep sadness, with the inner corners of the eyebrows raised, drooping upper eyelids, and the corners of the lips pulled downward, often in a somber or isolated environment.",
    "A person captured in a moment of sudden surprise, with eyes wide open, eyebrows raised high and curved, and the jaw dropped open in astonishment, reacting to an unexpected event."
]

prompt_ensemble_caer = [
    [ # Anger
        "A photo of a person showing intense anger in their environment.",
        "A face with furrowed brows, a glare, and tight lips indicating rage.",
        "A deeply angry and hostile facial expression in context.",
        "A person visibly furious, with tense muscles and a scowling face.",
        "An expression of extreme frustration and anger."
    ],
    [ # Disgust
        "A photo of a person showing strong disgust towards their surroundings.",
        "A face with a severely wrinkled nose and raised upper lip.",
        "A disgusted and repulsed facial expression.",
        "A person looking nauseated or offended by something in the scene.",
        "An expression of intense aversion and disgust."
    ],
    [ # Fear
        "A photo of a person showing genuine fear in a threatening context.",
        "A face with wide, terrified eyes and raised eyebrows.",
        "A fearful, scared, and alarmed facial expression.",
        "A person looking extremely frightened and anxious.",
        "An expression of pure terror and panic."
    ],
    [ # Happy
        "A photo of a person showing pure happiness and joy.",
        "A brightly smiling face with raised cheeks and crinkled eyes.",
        "A delighted and joyful facial expression in a positive scene.",
        "A person laughing or grinning broadly with genuine happiness.",
        "An expression of warmth, cheerfulness, and delight."
    ],
    [ # Neutral
        "A photo of a person showing a completely neutral and calm expression.",
        "A relaxed face with no strong emotion or tension.",
        "A neutral, indifferent, and steady facial expression.",
        "A person looking perfectly calm in an ordinary everyday scene.",
        "An expression devoid of any specific emotional reaction."
    ],
    [ # Sad
        "A photo of a person showing deep sadness and sorrow.",
        "A face with a frown, drooping eyelids, and downturned lips.",
        "A sad, depressed, and melancholic facial expression.",
        "A person looking heartbroken or grieving in a somber context.",
        "An expression of misery, crying, or silent sorrow."
    ],
    [ # Surprise
        "A photo of a person showing sudden surprise or astonishment.",
        "A face with an open mouth, dropped jaw, and wide eyes.",
        "A surprised, shocked, and amazed facial expression.",
        "A person reacting to something unexpected in the scene.",
        "An expression of disbelief and sudden realization."
    ]
]


# EMOTIC Classes (26 classes)
class_names_emotic = [
    'Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 
    'Disapproval', 'Disconnection', 'Disquietment', 'Doubt/Confusion', 'Embarrassment', 
    'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear', 'Happiness', 'Pain', 
    'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 
    'Sympathy', 'Yearning'
]

class_names_with_context_emotic = [f"A person shows {c.lower()}." for c in class_names_emotic]

class_descriptor_emotic = [
    # 0. Affection — co-occurs with: Happiness, Pleasure
    "A person showing warm affection with gentle touch, a tender embrace, or a loving smile directed at someone nearby.",
    # 1. Anger — co-occurs with: Annoyance, Disapproval
    "A person displaying intense anger with furrowed brows, clenched fists, a glaring stare, and tense rigid body in a confrontational scene.",
    # 2. Annoyance — co-occurs with: Anger, Disapproval
    "A person looking bothered and irritated, with a slight frown, crossed arms, and an impatient stance in an uncomfortable situation.",
    # 3. Anticipation — co-occurs with: Engagement, Excitement, Confidence
    "A person actively waiting or expecting something, leaning forward with alert eyes, focused gaze, and an eager posture.",
    # 4. Aversion — co-occurs with: Disgust, Annoyance
    "A person feeling strong dislike or disgust, turning away with a wrinkled nose, repulsed expression, and avoidance body language.",
    # 5. Confidence — co-occurs with: Engagement, Excitement, Esteem
    "A person looking self-assured and confident, standing tall with open posture, direct eye contact, and a composed expression.",
    # 6. Disapproval — co-occurs with: Anger, Annoyance
    "A person expressing objection or disapproval, with a stern look, shaking head, pursed lips, and a critical stance.",
    # 7. Disconnection — co-occurs with: Fatigue, Sadness
    "A person looking isolated and disconnected, gazing away from others, with a withdrawn posture and blank expression in a social context.",
    # 8. Disquietment — co-occurs with: Fear, Doubt/Confusion
    "A person feeling restless and uneasy, fidgeting, with widened eyes, furrowed brows, and anxious body movements in an uncertain setting.",
    # 9. Doubt/Confusion — co-occurs with: Disquietment, Embarrassment
    "A person looking confused and uncertain, tilting their head, squinting eyes, with a puzzled frown and hesitant posture.",
    # 10. Embarrassment — co-occurs with: Doubt/Confusion, Sensitivity
    "A person feeling self-conscious and embarrassed, covering their face, looking down, blushing, or hiding behind others.",
    # 11. Engagement — co-occurs with: Anticipation, Excitement, Confidence
    "A person highly focused and mentally engaged, with alert eyes fixed on an activity, attentive posture, and concentrated expression.",
    # 12. Esteem — co-occurs with: Confidence, Happiness
    "A person feeling respected or showing pride, with a dignified posture, slight smile, and an air of accomplishment or recognition.",
    # 13. Excitement — co-occurs with: Engagement, Happiness, Anticipation
    "A person feeling enthusiastic and excited, with wide eyes, an energetic smile, animated gestures, and dynamic body movement.",
    # 14. Fatigue — co-occurs with: Disconnection, Sadness
    "A person looking exhausted and tired, with drooping eyelids, slouched posture, yawning, and low energy in their movements.",
    # 15. Fear — co-occurs with: Disquietment, Suffering
    "A person showing genuine fear or panic, with wide terrified eyes, raised eyebrows, open mouth, and a defensive or frozen posture.",
    # 16. Happiness — co-occurs with: Excitement, Pleasure, Engagement
    "A person experiencing joy and happiness, with a bright genuine smile, crinkled eyes, relaxed open posture, in a positive scene.",
    # 17. Pain — co-occurs with: Suffering, Sadness
    "A person experiencing physical or emotional pain, grimacing, clutching a body part, with a contorted expression and tense muscles.",
    # 18. Peace — co-occurs with: Pleasure, Happiness
    "A person feeling calm and at peace, with a serene expression, relaxed shoulders, soft eyes, and a tranquil still posture.",
    # 19. Pleasure — co-occurs with: Happiness, Engagement
    "A person experiencing enjoyment and pleasure, with a content smile, relaxed body, and a satisfied look while engaged in an activity.",
    # 20. Sadness — co-occurs with: Suffering, Pain, Disconnection
    "A person expressing sorrow and sadness, with downturned lips, watery eyes, slumped shoulders, and a withdrawn demeanor.",
    # 21. Sensitivity — co-occurs with: Sympathy, Embarrassment
    "A person looking emotionally sensitive and vulnerable, with a soft trembling expression, moist eyes, and a fragile posture.",
    # 22. Suffering — co-occurs with: Pain, Sadness, Fear
    "A person enduring hardship or suffering, with a pained grimace, hunched body, and visible distress in a difficult context.",
    # 23. Surprise — co-occurs with: Fear, Excitement
    "A person looking astonished or surprised, with wide open eyes, raised eyebrows, dropped jaw, and a sudden backward lean.",
    # 24. Sympathy — co-occurs with: Affection, Sensitivity
    "A person feeling compassion and sympathy, with a caring gaze, gentle touch toward someone, and an empathetic expression.",
    # 25. Yearning — co-occurs with: Sadness, Anticipation
    "A person showing strong desire or yearning, with a wistful gaze, reaching gesture, and a longing expression directed at something distant."
]

prompt_ensemble_emotic = [
    [  # 0. Affection (co-occurs: Happiness, Pleasure)
        "A person showing warm affection with a tender embrace or loving touch.",
        "A person with a gentle caring smile reaching out to comfort someone.",
        "A scene of emotional warmth between people sharing a loving moment.",
        "A person displaying fondness with soft eyes and open welcoming arms.",
        "A close interaction between people showing love and emotional closeness."
    ],
    [  # 1. Anger (co-occurs: Annoyance, Disapproval)
        "A person displaying intense anger with furrowed brows and clenched fists.",
        "A person with a hostile glare, rigid body, and aggressive confrontational stance.",
        "A scene of conflict showing a person seething with rage and frustration.",
        "A person shouting or arguing with visible anger in their expression.",
        "A furious person with tense muscles and a threatening aggressive posture."
    ],
    [  # 2. Annoyance (co-occurs: Anger, Disapproval)
        "A person looking bothered and irritated with a slight frown.",
        "A person with crossed arms and an impatient expression of mild frustration.",
        "A scene showing someone visibly annoyed by something in their surroundings.",
        "A person rolling their eyes or sighing in annoyance.",
        "A mildly frustrated person showing displeasure without full-blown anger."
    ],
    [  # 3. Anticipation (co-occurs: Engagement, Excitement, Confidence)
        "A person eagerly waiting and expecting something with focused attention.",
        "A person leaning forward with alert eyes and an anticipatory expression.",
        "A scene showing someone poised and ready for an upcoming event or action.",
        "A person with widened eyes and slightly parted lips in keen anticipation.",
        "A person watching intently with excitement about what is coming next."
    ],
    [  # 4. Aversion (co-occurs: Disgust, Annoyance)
        "A person feeling strong dislike, turning away with a wrinkled nose.",
        "A person with a repulsed expression, leaning back from something unpleasant.",
        "A scene showing someone avoiding or recoiling from an unwanted stimulus.",
        "A person displaying physical avoidance with disgust on their face.",
        "A person shielding themselves from something they find deeply unpleasant."
    ],
    [  # 5. Confidence (co-occurs: Engagement, Excitement, Esteem)
        "A person looking self-assured with a tall upright posture and direct gaze.",
        "A person standing confidently with open body language and a composed expression.",
        "A scene showing someone in control, leading, or performing with confidence.",
        "A person with a slight assured smile and relaxed shoulders showing self-belief.",
        "A person demonstrating mastery and poise in their actions and demeanor."
    ],
    [  # 6. Disapproval (co-occurs: Anger, Annoyance)
        "A person expressing disapproval with a stern critical look and pursed lips.",
        "A person shaking their head with a judgmental frown of disagreement.",
        "A scene showing someone objecting to or criticizing something around them.",
        "A person with crossed arms and a skeptical disapproving expression.",
        "A person looking on with clear dissatisfaction and a negative assessment."
    ],
    [  # 7. Disconnection (co-occurs: Fatigue, Sadness)
        "A person looking isolated and disconnected, gazing away from everyone.",
        "A person with a blank withdrawn expression, disengaged from the surroundings.",
        "A scene showing someone alone and emotionally detached from the group.",
        "A person staring into space with an absent and uninvolved demeanor.",
        "A person physically present but mentally elsewhere, showing emotional distance."
    ],
    [  # 8. Disquietment (co-occurs: Fear, Doubt/Confusion)
        "A person feeling restless and uneasy with nervous fidgeting.",
        "A person with widened eyes, furrowed brows, and anxious body movements.",
        "A scene showing someone uncomfortable and troubled by their surroundings.",
        "A person biting their lip or wringing hands in visible unease.",
        "A person looking around nervously with a worried unsettled expression."
    ],
    [  # 9. Doubt/Confusion (co-occurs: Disquietment, Embarrassment)
        "A person looking confused with a tilted head and puzzled frown.",
        "A person squinting their eyes with uncertainty and a hesitant posture.",
        "A scene showing someone struggling to understand or make a decision.",
        "A person scratching their head with a bewildered expression.",
        "A person looking back and forth with indecision and visible confusion."
    ],
    [  # 10. Embarrassment (co-occurs: Doubt/Confusion, Sensitivity)
        "A person feeling embarrassed, covering their face and looking down.",
        "A person blushing or hiding behind something with a self-conscious expression.",
        "A scene showing someone in an awkward situation feeling exposed.",
        "A person averting their gaze with a sheepish uncomfortable smile.",
        "A person shrinking away from attention in visible embarrassment."
    ],
    [  # 11. Engagement (co-occurs: Anticipation, Excitement, Confidence)
        "A person deeply focused and mentally absorbed in an activity.",
        "A person with alert attentive eyes fixed on their task with concentration.",
        "A scene showing someone fully immersed and participating actively.",
        "A person leaning in with engaged body language and intense focus.",
        "A person completely absorbed in what they are watching or doing."
    ],
    [  # 12. Esteem (co-occurs: Confidence, Happiness)
        "A person feeling proud and respected with a dignified bearing.",
        "A person standing with self-respect and an air of accomplishment.",
        "A scene showing someone receiving recognition or feeling valued.",
        "A person with a proud upright posture and a satisfied confident look.",
        "A person radiating a sense of worth and achievement."
    ],
    [  # 13. Excitement (co-occurs: Engagement, Happiness, Anticipation)
        "A person feeling excited with wide eyes, big smile, and energetic gestures.",
        "A person jumping, cheering, or clapping with enthusiasm and joy.",
        "A scene showing someone thrilled and animated about an event.",
        "A person with dynamic body movement expressing pure excitement.",
        "A person radiating enthusiasm with an electric energized expression."
    ],
    [  # 14. Fatigue (co-occurs: Disconnection, Sadness)
        "A person looking exhausted with drooping eyelids and slouched posture.",
        "A person yawning or rubbing their eyes with visible tiredness.",
        "A scene showing someone drained of energy, barely keeping their eyes open.",
        "A person with a heavy head resting on their hand in fatigue.",
        "A person moving slowly with low energy and a weary expression."
    ],
    [  # 15. Fear (co-occurs: Disquietment, Suffering)
        "A person showing fear with wide terrified eyes and a frozen defensive posture.",
        "A person backing away in panic with raised eyebrows and open mouth.",
        "A scene showing someone in danger or facing a threatening situation.",
        "A person trembling or cowering with genuine terror on their face.",
        "A person screaming or gasping in a moment of pure fright."
    ],
    [  # 16. Happiness (co-occurs: Excitement, Pleasure, Engagement)
        "A person experiencing joy with a bright genuine smile and crinkled eyes.",
        "A person laughing heartily with a relaxed happy expression.",
        "A scene of celebration or fun showing a person radiating happiness.",
        "A person with a warm beaming smile in a positive joyful context.",
        "A person showing pure delight and contentment in their expression."
    ],
    [  # 17. Pain (co-occurs: Suffering, Sadness)
        "A person experiencing pain with a grimace and contorted expression.",
        "A person clutching a body part or wincing in visible physical distress.",
        "A scene showing someone hurt and in obvious physical or emotional pain.",
        "A person with clenched teeth and tense muscles from pain.",
        "A person crying out or doubling over in acute discomfort."
    ],
    [  # 18. Peace (co-occurs: Pleasure, Happiness)
        "A person feeling calm and peaceful with a serene relaxed expression.",
        "A person with soft eyes and a gentle smile in a tranquil setting.",
        "A scene of stillness showing someone at complete ease and rest.",
        "A person meditating or resting quietly with a composed demeanor.",
        "A person in a natural calm environment looking deeply relaxed."
    ],
    [  # 19. Pleasure (co-occurs: Happiness, Engagement)
        "A person experiencing pleasure with a content satisfied smile.",
        "A person savoring an enjoyable moment with relaxed body language.",
        "A scene showing someone delighting in a pleasant activity or experience.",
        "A person with a look of enjoyment and deep satisfaction.",
        "A person indulging in something with visible delight and comfort."
    ],
    [  # 20. Sadness (co-occurs: Suffering, Pain, Disconnection)
        "A person expressing sorrow with downturned lips and watery eyes.",
        "A person crying or on the verge of tears with a dejected posture.",
        "A scene showing someone grieving or mourning in a somber setting.",
        "A person with slumped shoulders and a heavy sorrowful expression.",
        "A person sitting alone looking down with deep melancholy."
    ],
    [  # 21. Sensitivity (co-occurs: Sympathy, Embarrassment)
        "A person looking emotionally sensitive with a soft vulnerable expression.",
        "A person with moist eyes and a trembling lip showing emotional fragility.",
        "A scene showing someone deeply moved and emotionally touched.",
        "A person reacting with heightened emotional sensitivity to something.",
        "A person tearing up or looking deeply affected by an emotional moment."
    ],
    [  # 22. Suffering (co-occurs: Pain, Sadness, Fear)
        "A person enduring suffering with a pained grimace and hunched body.",
        "A person in visible distress from hardship or emotional anguish.",
        "A scene showing someone struggling through a difficult painful situation.",
        "A person bearing pain or grief with a tortured expression.",
        "A person overwhelmed by suffering, looking broken and exhausted."
    ],
    [  # 23. Surprise (co-occurs: Fear, Excitement)
        "A person looking surprised with wide open eyes and dropped jaw.",
        "A person with raised eyebrows and a sudden backward lean of astonishment.",
        "A scene showing someone caught off guard by an unexpected event.",
        "A person gasping with hands raised in a moment of shock.",
        "A person with a startled expression reacting to something unforeseen."
    ],
    [  # 24. Sympathy (co-occurs: Affection, Sensitivity)
        "A person feeling compassion with a caring empathetic expression.",
        "A person gently touching or comforting someone in distress.",
        "A scene showing someone listening with deep sympathy and concern.",
        "A person with a soft warm gaze directed at someone who is suffering.",
        "A person showing solidarity and emotional support through their demeanor."
    ],
    [  # 25. Yearning (co-occurs: Sadness, Anticipation)
        "A person showing yearning with a wistful gaze into the distance.",
        "A person reaching out or looking longingly at something far away.",
        "A scene showing someone deeply wanting something they cannot have.",
        "A person with a nostalgic expression and a sense of unfulfilled desire.",
        "A person looking through a window with longing and deep want."
    ]
]


class_names_daisee = ['Very Low', 'Low', 'High', 'Very High']

class_names_with_context_daisee = [
    "A student shows very low engagement.",
    "A student shows low engagement.",
    "A student shows high engagement.",
    "A student shows very high engagement."
]

class_descriptor_daisee = [
    "A student is completely disengaged, looking away, sleeping, or doing something else entirely.",
    "A student is distracted, frequently looking around, yawning, or showing little interest.",
    "A student is paying attention, looking at the screen, and following the lesson.",
    "A student is highly focused, leaning forward, taking notes, and reacting to the content."
]

prompt_ensemble_daisee = [
    [ # Very Low (0)
        "A video of a student with very low engagement.",
        "A student looking away or sleeping.",
        "A completely disengaged student."
    ],
    [ # Low (1)
        "A video of a student with low engagement.",
        "A student looking distracted or bored.",
        "A student showing little interest in the lesson."
    ],
    [ # High (2)
        "A video of a student with high engagement.",
        "A student looking at the screen attentively.",
        "A student following the lecture."
    ],
    [ # Very High (3)
        "A video of a highly engaged student.",
        "A student leaning forward and taking notes.",
        "A student completely absorbed in learning."
    ]
]
