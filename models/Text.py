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
    "A person feeling fond and loving.",
    "A person displaying intense anger and rage.",
    "A person looking bothered and irritated.",
    "A person expecting or waiting for something.",
    "A person feeling disgust or strong dislike.",
    "A person looking self-assured and confident.",
    "A person expressing objection or disapproval.",
    "A person looking isolated or disconnected from the surroundings.",
    "A person feeling restless, uneasy, or anxious.",
    "A person looking confused, uncertain, or doubtful.",
    "A person feeling self-conscious or embarrassed.",
    "A person highly focused and engaged in an activity.",
    "A person feeling respected or showing self-esteem.",
    "A person feeling enthusiastic and excited.",
    "A person looking exhausted, tired, or fatigued.",
    "A person showing genuine fear or panic.",
    "A person experiencing joy and happiness.",
    "A person experiencing physical or emotional pain.",
    "A person feeling calm, relaxed, and at peace.",
    "A person experiencing enjoyment and pleasure.",
    "A person expressing sorrow and sadness.",
    "A person looking sensitive or emotionally vulnerable.",
    "A person experiencing hardship or suffering.",
    "A person looking astonished or surprised.",
    "A person feeling compassion and sympathy.",
    "A person showing strong desire or yearning."
]

prompt_ensemble_emotic = [
    [f"A photo of a person showing {c.lower()}.", f"A person feeling {c.lower()}.", f"An expression of {c.lower()}."]
    for c in class_names_emotic
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
