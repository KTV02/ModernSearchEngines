
def check_keywords_in_document(document_path, keywords, encoding='utf-8'):
    """
    Checks if a document contains any of the given keywords and prints out which keywords are found.

    Args:
        document_path (str): The path to the document file.
        keywords (list): A list of keywords to check in the document.
        encoding (str): The encoding of the document file.
    """
    try:
        with open(document_path, 'r', encoding=encoding) as file:
            content = file.read()
        
        found_keywords = [keyword for keyword in keywords if keyword in content]
        
        if found_keywords:
            print("The document contains the following keywords:")
            for keyword in found_keywords:
                print(f"- {keyword}")
        else:
            print("No keywords found in the document.")
    
    except FileNotFoundError:
        print(f"Error: The file '{document_path}' was not found.")
    except UnicodeDecodeError as e:
        print(f"Unicode decode error: {e}. Try a different encoding.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
document_path = 'website.txt'
keywords = ['tübingen', 'tubingen', 'tuebingen', 'tuebing', 'tübing', 't%c3%bcbingen', 'wurmlingen', 'wolfenhausen', 'wilhelmshöhe', 'wendelsheim', 'weitenburg', 'weilheim', 'wankheim', 'waldhörnle', 'waldhausen', 'wachendorf', 'unterjesingen', 'landkreis tübingen', 'tübingen', 'talheim', 'sulzau', 'sülchen', 'streimberg', 'stockach', 'westliche steingart', 'steinenberg', 'seebronn', 'schwärzloch', 'schwalldorf', 'schönbuchspitz', 'naturpark schönbuch', 'schönberger kopf', 'schloßlesberg', 'schloßbuckel', 'schadenweilerhof', 'saurücken', 'rottenburg', 'rosenau', 'reusten', 'remmingsheim', 'rappenberg', 'poltringen', 'pfrondorf', 'pfäffingen', 'pfaffenberg', 'österberg', 'öschingen', 'ofterdinger berg', 'ofterdingen', 'odenburg', 'oberwörthaus', 'oberndorf', 'obernau', 'oberhausen', 'neuhaus', 'nellingsheim', 'nehren', 'mössingen', 'mähringen', 'lustnau', 'lausbühl', 'kusterdingen', 'kreuzberg', 'kreßbach', 'kirchkopf', 'kirchentellinsfurt', 'kilchberg', 'kiebingen', 'jettenburg', 'immenhausen', 'hornkopf', 'horn', 'hohenstöffel', 'schloss hohenentringen', 'hochburg', 'hirschkopf', 'hirschau', 'hirrlingen', 'hinterweiler', 'heubergerhof', 'heuberg', 'heuberg', 'hennental', 'hemmendorf', 'härtlesberg', 'hailfingen', 'hagelloch', 'günzberg', 'gomaringen', 'geißhalde', 'galgenberg', 'frommenhausen', 'firstberg', 'filsenberg', 'felldorf', 'farrenberg', 'bahnhof eyach', 'ergenzingen', 'erdmannsbach', 'ammerbuch', 'einsiedel', 'eichenfirst', 'eichenberg', 'ehingen', 'eckenweiler', 'höhe', 'dußlingen', 'dürrenberg', 'dickenberg', 'dettingen', 'dettenhausen', 'derendingen', 'denzenberg', 'buß', 'burg', 'buhlbachsaue', 'bühl', 'bühl', 'bühl', 'bromberg', 'breitenholz', 'börstingen', 'bodelshausen', 'bläsiberg', 'bläsibad', 'bierlingen', 'bieringen', 'belsen', 'bei der zeitungseiche', 'bebenhausen', 'baisingen', 'bad sebastiansweiler', 'bad niedernau', 'ammern', 'ammerbuch', 'altstadt', 'altingen', 'alter berg', 'flugplatz poltringen ammerbuch', 'starzach', 'neustetten', 'hotel krone tubingen', 'hotel katharina garni', 'bodelshausen', 'dettenhausen', 'dußlingen', 'gomaringen', 'hirrlingen', 'kirchentellinsfurt', 'kusterdingen', 'nehren', 'ofterdingen', 'mössingen', 'rottenburg am neckar', 'tübingen, universitätsstadt', 'golfclub schloß weitenburg', 'siebenlinden', 'steinenbertturm', 'best western hotel convita', 'bebenhausen abbey', 'schloss bebenhausen', 'burgstall', 'rafnachberg', 'östliche steingart', 'kirnberg', 'burgstall', 'großer spitzberg', 'kleiner spitzberg', 'kapellenberg', 'tannenrain', 'grabhügel', 'hemmendörfer käpfle', 'kornberg', 'rotenberg', 'weilerburg', 'martinsberg', 'eckberg', 'entringen', 'ofterdingen, rathaus', 'randelrain', 'wahlhau', 'unnamed point', 'spundgraben', 'university library tübingen', 'tübingen hbf', 'bad niedernau', 'bieringen', 'kiebingen', 'unterjesingen mitte', 'unterjesingen sandäcker', 'entringen', 'ergenzingen', 'kirchentellinsfurt', 'mössingen', 'pfäffingen', 'rottenburg (neckar)', 'tübingen west', 'tübingen-lustnau', 'altingen (württ)', 'bad sebastiansweiler-belsen', 'dußlingen', 'bodelshausen', 'nehren', 'tübingen-derendingen', 'dettenhausen']

check_keywords_in_document(document_path, keywords)
