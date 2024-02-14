from ukrainian_word_stress import Stressifier, StressSymbol
import os
import pandas as pd
import csv

stressify = Stressifier(stress_symbol="ˈ")

# Step 1: List all files in the folder
folder_path = 'datasets/elevenlabs_dataset/'

# Step 2: Read the CSV file
csv_path = 'datasets/elevenlabs_dataset/filelists/ljs_audio_text_val_filelist.txt'
result_csv_path = 'datasets/elevenlabs_dataset/filelists/_ljs_audio_text_val_filelist.txt'

unstressed = ["антикваріату", "альтернативні", "працівників", "проаналізовані", "п'ятдесяти", "організувати", "маніпулювати", "мінімізувати", "університетського", "автоматизовані", "альтернативного", "контролювати", "неприпустимо", "багаторазові", "компенсувати", "організував", "бетономішалкою", "прогнозуванні", "подорожувати", "бездротовим", "оптимізувати", "спостерігав", "ґрунтообробну", "чотирьохсот", "правоохоронних", "колекціонерів", "рекультивації", "найголовніше", "програмування", "червонокнижний", "холоднокровність", "бездротових", "гомогенізованому", "спеціалізуються", "відпочинковий", "спостерігає", "університету", "фундаментальних", "інопланетянами", "монополізації", "контролюватимуть", "найрозумніша", "автоматизації", "синхронізується", "університети", "дезінформацію", "розшифрувати", "катастрофічних", "телефонувати", "відреагувати", "середньовіччі", "біотехнології", "паралізувавши"]
stressed = ["антикваріаˈту", "альтернатиˈвні", "працівникіˈв", "проаналізоˈвані", "п'ятдесятиˈ", "організуваˈти", "маніпулюваˈти", "мінімізуваˈти", "університеˈтського", "автоматизоˈвані", "альтернатиˈвного", "контролюваˈти", "неприпустиˈмо", "багатораˈзові", "компенсуваˈти", "організуваˈв", "бетономішаˈлкою", "прогнозуваˈнні", "подорожуваˈти", "бездротовиˈм", "оптимізуваˈти", "спостерігаˈв", "ґрунтооброˈбну", "чотирьохсоˈт", "правоохороˈнних", "колекціонеˈрів", "рекультиваˈції", "найголовніˈше", "програмуваˈння", "червонокниˈжний", "холоднокроˈвність", "бездротовиˈх", "гомогенізоˈваному", "спеціалізуˈються", "відпочинкоˈвий", "спостерігаˈє", "університеˈту", "фундаментаˈльних", "інопланетяˈнами", "монополізаˈції", "контролюваˈтимуть", "найрозумніˈша", "автоматизаˈції", "синхронізуˈється", "університеˈти", "дезінформаˈцію", "розшифруваˈти", "катастрофіˈчних", "телефонуваˈти", "відреагуваˈти", "середньовіˈччі", "біотехнолоˈгії", "паралізуваˈвши"]

with open(csv_path, 'r', newline='') as file:
    reader = list(csv.reader(file, delimiter='|'))
    with open(result_csv_path, 'w', newline='') as file:
        for row in reader:
            stressified_text = stressify(row[1])
            for word in unstressed:
                if word in stressified_text:
                    stressified_text = stressified_text.replace(word, stressed[unstressed.index(word)])
            file.write(f"{row[0]}|{stressified_text}\n")
