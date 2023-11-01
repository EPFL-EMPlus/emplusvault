import os
import pandas as pd
import fnmatch
from typing import Optional, List, Dict, Any
from pathlib import Path
from lxml import etree

import rts.utils

LOG = rts.utils.get_logger()

DATA_KEYS = [
        'c_theme_en', # 404165 rows
        'f_action_en', # 160566 rows
        'f_mouvement_en', # 117245 rows
        'f_emotion_en', # 22503 rows
        'f_valeur_en', # 1459 rows
        'f_symbole_en', # 1229 rows
        'f_symbole_en_189',
        'f_symbole_en_191',
        'c_epreuve_en', # None
        'f_principe_en', # None
        'medaille_en',  # None
        'medaille_en_133', 
        'medaille_en_135',
        'c_personnalite_en',  # None
        'olympien_en', # None
        'f_activite_autre_en', # None
        'f_activite_autre_en_111',
        'f_activite_autre_en_112',
        'f_activite_jeux_en', # None
        'f_lieux_ref_continent_en', # None
        'f_lieux_ref_continent_en_165', 
        'f_lieux_ref_pays_en', 
        'f_lieux_ref_pays_en_169',
        'f_lieux_ref_site_olympique_en',
        'f_lieux_ref_site_olympique_en_173',
        'f_lieux_ref_site_olympique_en_175',
        'f_lieux_ref_site_olympique_en_177',
        'f_lieux_ref_ville_en',
        'f_lieux_ref_ville_en_181',
        'f_lieux_ref_site_en',
        'f_lieux_ref_site_en_185',
        'f_sport_ete_en',  # 29 rows
        'f_sport_ete_en_119',
        'f_sport_ete_en_198',
        'f_sport_ete_en_199',
        'f_sport_hiver_en', # None
        'f_sport_hiver_en_123',
        'f_sport_hiver_en_202',
        'f_sport_hiver_en_203',
        'f_sport_autre_en', # None
        'f_sport_autre_en_127',
        'f_sport_autre_en_206',
        'f_sport_autre_en_207',
        'f_athlete_fr',
        'f_athlete_en',
        'f_lieux_ref_site_olympique_en_2',
        'c_cadrage_en',
        'f_team_en',
        'f_autre_decription_en',
]

SPORT_MAP = {
    'Aquatic Sports': ['Aquatic sports', 'Synchronized S.'],
    'Archery': ['Archery', 'Archery Teams', 'Archery Individual'],
    'Athletics': ['Athletics', 'Athletics decathlon', 'Athletics Javelin', 'High jump', 'Long jump', '800 m', 
                  'Javelin', 'Shot put', 'Athletics heptathlon 800 m', 
                  'Athletics heptathlon 200 m', 'Athletics 3000 m Steeplechase', 
                  'Athletics decathlon pole vault', 'Athletics heptathlon high jump', 
                  'Athletics decathlon long jump', 'Athletics decathlon 100 m'],
    'Badminton': ['Badminton', 'Badminton Double', 'Badminton Single'],
    'Beach Volleyball': ['Beach Volleyball', 'Beach volleyball', 'Beach volley.', 'Volleyball de plage'],
    'Basque Pelota': ['Basque Pelota'],
    'Baseball': ['Baseball', 'Softball'],
    'Basketball': ['Basketball', '3x3 Basketball'],
    'Boxing': ['Boxing'],
    'Canoeing': ['Canoeing', 'Canoe Slalom', 'Canoe Sprint', 'Canoe ', 'Canoe', 'Slalom Kayaking'],
    'Cycling': ['BMX', 'Cycling Track', 'Cycling Mountain Bike', 'Cycling Road', 'Cycling BMX Freestyle', 'Cycling Track Race, individual', 'Cycling BMX', 'Cycling BMX Racing'],
    'Diving': ['Diving'],
    'Equestrian': ['Equestrian ', 'Equestrian', 'Jumping', 'Dressage', 'Eventing'],
    'Fencing': ['Fencing', 'Fencing, individual', 'Fencing Foil, teams'],
    'Football': ['Football'],
    'Golf': ['Golf'],
    'Gymnastics': ['Gymnastics Multiple indiv. II Combined', 'Artistic Gymnastics', 'Gymnastics Artistic ', 'Artistic gymnastics', 'Artistic G.', 'Rhythmic Gymnastics', 'Rhythmic G.', 'Trampoline Gymnastics', 'Gymnastics', 'Artistic gymnastics floor exercises', 'Rhythmic gymnastics group competition', 'Trampoline'],
    'Handball': ['Handball'],
    'Hockey': ['Hockey'],
    'Judo': ['Judo'],
    'Karate': ['Karate'],
    'Modern Pentathlon': ['Modern Pentathlon', 'Modern Pentath.', 'Modern pentathlon Shooting', 'Modern Pentathlon - Fencing, individual'],
    'Roller Hockey': ['Rink-Hockey', 'Roller hockey'],
    'Rowing': ['Rowing', 'Rowing single sculls', 'Rowing Coxwain Eight in point', 'Rowing Coxwainless Pairs in point'],
    'Rugby': ['Rugby Sevens', ' Rugby Sevens', 'Rugby'],
    'Sailing': ['Sailing', 'Yachting Boardsailing'],
    'Shooting': ['Shooting', 'Tir', '10m cible mobile (30+30 coups)[H]', 'fosse olympique (125 cibles)[X]'],
    'Skateboarding': ['Skateboarding'],
    'Softball Centre': ['Softball Centre'],
    'Sport Climbing': ['Sport Climbing'],
    'Surfing': ['Surfing'],
    'Swimming': ['Synchronized Swimming', 'Swimming', 'Marathon Swimming', 'Swimming 100 m Butterfly', 'Swimming 400 m Freestyle', 'Swimming 400 m Medley', 'Swimming 100 m Breaststroke', 'Swimming 4 x 100 m Freestyle Relay', 'Swimming Freestyle', '200 m brasse femmes[F]', 'Artistic Swimming'],
    'Table Tennis': ['Table Tennis', 'Table tennis', 'Table tennis Single', 'Table tennis Double'],
    'Taekwondo': ['Taekwondo', 'Tae kwon do', 'Taekwondo 57 - 67 kg'],
    'Tennis': ['Tennis Double', 'Tennis Single', 'Tennis'],
    'Triathlon': ['Triathlon'],
    'Volleyball': ['Volleyball'],
    'Water Polo': ['Water polo', 'Water Polo'],
    'Weightlifting': ['Weightlifting', 'Weightlifting 63 - 69 kg'],
    'Wrestling': ['Wrestling Freestyle', 'Wrestling Free.', 'Wrestling Greco-Roman', 'Wrestling', 'Wrestling Gre-R'],
    'Wushu': ['Wushu']
}


NON_SPORTS = ['Volunteer', 'nan', 'Finales', 'Scott', 'PATRICK, Jaele', 
              'ARAKAWA, Eriko', 'http:', 'BERHANU, Dejene', 'DEITERS, Julie', '', 
              'MASLIVETS, Olha', 'Athletics decathlon discus', 'SCHLOESSER, Gabriela',
              'GREECE - Cycling Track', 'ABIR ABDELRAHMAN, Khalil Mahmoud K', 'Série 3']


ROUND_MAPPING = {
    'Preliminary': ['Preliminary', 'Qualification', 'Preliminary Round', 'Pool Matches', 'Heats',
                    'Preliminary Round Group B', 'Preliminary Round - Pool B', 'Preliminary - Pool A', 
                    'Pool B', 'Preliminary - Pool D', 'Preliminary Round - Group A', 
                    'Preliminary Round Group A', 'Qualifying', 'Preliminary Round - Pool A', 
                    'Preliminary Round - Group B', 'Preliminary - Pool E', 
                    'Preliminary - Pool C', 'Preliminary - Pool F', 'Preliminary - Pool B', 
                    'Preliminary Round Group C', 'Preliminary Round Group E', 
                    'Preliminary round Groupe E', 'Grand Prix - Qualifier', 'Jumping Qualifier', 
                    "Men's Light Heavy (81kg) Preliminary Round 2", 'Preliminaries - Round of 16'],

    'Quarterfinal': ['Quarterfinal', 'Quarterfinals', 'Round of 16', 'Quarter-finals', 
                     'Quarter final', 'Last 16', 'Elimination Round of 16', 'Round R. 17-20'],

    'Semifinal': ['Semifinals', 'Semifinal', 'Semi-finals', 'Semi-Final', 
                  'Semi final', 'Semi finals', 'Semi-final'],

    'Final': ['Final', 'Final Ranking', 'Gold Medal Contest', 
              'Finals', 'Final round', 'Final ranking', 
              'Final Round', 'Final group ', 'Final Repechage', 
              'Individual All-Around Final', 'final', 'Final Round'],

    'Medal Ceremony': ['Victory Ceremony', 'Bronze Medal Matches', 
                       'Medal Race', 'Race 10', 'Bronze Medal Contests', 
                       'Medal Bouts', 'Medal Race'],

    'Match for Placement': ['Classifications 5-8', 'Classification 5th-8th', 
                            'Placing 9-12', 'Placing 5-8', 'Matches for Bronze Medals'],

    'Repechage': ['Repechage', 'Repechages', 'Repechage Round 2', 
                  '1st round repec', '7th round - repechages group A', 
                  '6th round - repechages group A', 'Repechage Round 1', '2nd round repec'],

    'Other': ['High Jump', 'team, dressage test[X]', 'Cross Country', 'All Groups', 
              'Grand Prix Freestyle', 'Long Jump', 'Javelin Throw', 'K-2 500m(M)', 
              'Round 1', 'individual, dressage test[X]', 'match race keelboat open (Yngling)[X]', 
              'Qualifier', 'Grand prix - special', 'individual, cross country test[X]', 
              'match race keelboat open (Soling)[X]', 'team, jumping test[X]', 
              'Contests for Bronze Medals', 'Grand Prix Special', 
              'Part of event or sub-event', 'Pole Vault', 
              "Women's Foil Individual Pool round", 'Shot Put', 'C-1 (single)(M)', 
              'K-1 (single)(W)', '100m Hurdles', 'team, cross country test[X]', 
              "Women's Individual Stroke Play Round 1", 'Softball[W]', 'Dressage', 
              'Part of event', 'individual[X]', 'Pool', 'K-1 500m (kayak single)[M]', 
              'C-1 500m (canoe single)[M]', 'Round Robin', 'Ev. Ind. Dres.', 
              '4km Ind. Purs.', 'Tempo Race 2', 
              'Elimination Race 3', 'Swimming 200m Freestyle', "Men's Light (60kg) Finals", 
              "Men's Épée Individual Pool round", '5th round - rep', 
              'K-2 500m (kayak double)[M]', 'Laser Run', 'Race 15', 'Seeding phase', 
              '100m', '100 m', 'Jumping', 
              'Mixed Doubles First Round', "Women's Riding Show Jumping", 
              ' Cross-Country[W]', 'www.cio-pam.org', '30km Pnt. Race', '20km Pnt Race', 
              'Seeding', 'K-1 1000m(M)', 'v', "Women's Individual Stroke Play Round 2", 
              '3km Ind. Purs.']
}


def read_all_xml_files(directory: str):
    xml_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if fnmatch.fnmatch(file, '*.xml'):
                xml_files.append(os.path.join(root, file))
    
    return sorted(xml_files)


def parse_data_mapping(filepath: str) -> Dict:
    tree = etree.parse(filepath) 
    fields = tree.xpath('//field')

    res = {}
    # loop over the fields
    for i, field in enumerate(fields):
        # get the column name
        column = field.get("column")
        # check if column ends with "_en"
        if column.endswith("_fr"):
            continue
        # get the xpath value
        xpath = field.get("xpath")
        xpath = xpath.replace("/DATASET/", "", 1)

        if column in res:
            column = column + f"_{i}"
        res[column] = xpath

    return res


def filter_mapping(mapping: Dict, columns: List[str] = DATA_KEYS) -> Dict:
    return {key: value for key, value in mapping.items() if key in columns}


def extra_metadata_from_mapping(item: Any, mapping: Dict) -> Dict:
    if not mapping:
        return {}
    
    parsed_dict = {key: [] for key in mapping}
    for key, value in mapping.items():
        try:
            s = ".//{}".format(value)
            # print(s)
            for elem in item.xpath(s):
                if isinstance(elem, etree._ElementUnicodeResult):
                    parsed_dict[key].append(elem)
                elif isinstance(elem, etree._Element):
                    parsed_dict[key].append(elem.text)
        except:
            pass

    parsed_dict = {key: value for key, value in parsed_dict.items() if value not in ['', [], None]}
    return parsed_dict


def parse_xml_with_mapping(filepath: str, mapping: Dict) -> List[Dict]:
    LOG.debug(f'Parsing {filepath}')
    tree = etree.parse(filepath)
    root = tree.getroot()

    seq = []
    for item in root.iter('Item'):
        parsed_dict = extra_metadata_from_mapping(item, mapping)
        seq.append(parsed_dict)
    return seq


def correct_timestamp_format(timestamp):
    # Split the timestamp at the last colon
    parts = timestamp.rsplit(':', 1)
    
    # Join the parts with a period
    corrected_timestamp = '.'.join(parts)    
    return corrected_timestamp


def parse_xml(filepath: str, mapping: Optional[Dict] = None) -> List[Dict]:
    LOG.debug(f'Parsing {filepath}')
    # parse XML
    tree = etree.parse(filepath)
    root = tree.getroot()

    # get the global GuidMedia
    global_guid_media = root.get('GuidMedia')

    # initialize list to hold sequences
    sequences = []
    global_desc = ''

    # iterate over sequences
    for item in root.xpath('.//Item'):
        sequence = {}
        sequence['guid'] = global_guid_media
        sequence['seq_id'] = item.get('GuidMedia')

        dd = item.xpath('.//E70/P94/E65/P4')[0] if item.xpath('.//E70/P94/E65/P4') else None
        if dd is not None:
            sequence['date'] = dd.get('d1')
        sequence['duration'] = item.xpath('.//E70/P3.dimensions/E62')[0].text if item.xpath('.//E70/P3.dimensions/E62') else None

        # start and end timestamps
        timestamps = item.xpath('.//E70/P43/E54/P90.value')
        sequence['start'] = correct_timestamp_format(timestamps[0].text) if timestamps else None
        sequence['end'] = correct_timestamp_format(timestamps[1].text) if timestamps else None

        is_public = item.xpath('.//E70')[0].get('P2.diffusion') == "PUBLIC" if item.xpath('.//E70') else None
        sequence['public'] = is_public

        # main_title in English
        titles = item.xpath('.//E70/P102/E35/E62')
        for title in titles:
            if title.get('lang') == 'en':
                sequence['description'] = title.text

        # P2.subtype description in English
        subtypes = item.xpath('.//E70/P67/E7/P10/E7/P2.subtype')
        for subtype in subtypes:
            sequence['event'] = subtype.get('en')
            # sequence['event_type'] = subtype.get('classe')

        # if not global_desc:
        #     global_desc = item.xpath('.//E70/E70_TERME_EN')[0].text if item.xpath('.//E70/E70_TERME_EN') else None

        extra = extra_metadata_from_mapping(item, mapping)
        sequence['extra'] = extra
        sequences.append(sequence)

    return sequences


def parse_all_xml_files(directory: str, mapping: Optional[Dict] = None) -> List[Dict]:
    xml_files = read_all_xml_files(directory)
    results = []

    for file in xml_files:
        results += parse_xml(file, mapping)

    return results


def items_to_dataframe(items: List[Dict]) -> pd.DataFrame:
    def split_time(x: str) -> pd.Timedelta:
        if not x:
            return pd.Timedelta()
        else:
            toks = x.split(':')
            t = pd.Timedelta(hours=int(toks[0]), 
                         minutes=int(toks[1]),
                         seconds=int(toks[2]),
                         milliseconds=int(toks[3]) if len(toks) > 3 else 0)
            return t

    df = pd.DataFrame.from_records(items)
    df = df[df.public]  # keep only public sequences
    df = df[df.date > '1900-01-01'] 
    df = df.drop('public', axis=1)
    # df = df.astype(str)
    df['date'] = pd.to_datetime(df['date'])
    df.loc[:, 'duration'] = df['duration'].apply(split_time)
    df['duration_sec'] = df.duration.apply(lambda x: float(x.total_seconds()))
    return df


def clean_sports(df: pd.DataFrame) -> pd.DataFrame:
    reverse_mapping = {v: k for k, values in SPORT_MAP.items() for v in values}
    for item in NON_SPORTS:
        reverse_mapping[item] = 'Non-Sport'
    df['sport'] = df['sport'].map(reverse_mapping)
    df['sport'] = df['sport'].fillna('Non-Sport')
    
    return df


def clean_round(df: pd.DataFrame) -> pd.DataFrame:
    reverse_mapping = {v: k for k, values in ROUND_MAPPING.items() for v in values}
    df['round'] = df['raw_round'].map(reverse_mapping)
    df['round'].replace('nan', '', inplace=True)
    df['round'].fillna('', inplace=True)
    return df 


def process_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df['event'] = df['event'].fillna('')
    event = df['event'].str.split('/', expand=True)
    event = event.fillna('')
    event['details'] = event[3] + '/' + event[4] + '/' + event[5]
    event['details'].replace('//', '', regex=True, inplace=True)
    event.rename({0: 'sport',
                1: 'category',
                2: 'raw_round'}, axis=1, inplace=True)
    event.drop(columns=[3,4,5], inplace=True)

    event = clean_sports(event)
    event = clean_round(event)
    return df.join(event)


def get_sport_df(df: pd.DataFrame) -> pd.DataFrame:
    return df[~(df['sport'] == 'Non-Sport')]


def get_extra_metadata_df(df: pd.DataFrame) -> pd.DataFrame:
    return df[df['extra'].apply(lambda x: bool(x))]


def get_extra_prop_df(df: pd.DataFrame, prop: str) -> pd.DataFrame:
    # return df[df['extra'].apply(lambda x: prop in x)]
    return df[df['extra'].apply(lambda x: any(k.startswith(prop) for k in x.keys()) if isinstance(x, dict) else False)]


def get_extra_prop_series(df: pd.DataFrame, prop: str) -> pd.Series:
    ex = get_extra_prop_df(df, prop)['extra']
    return ex.apply(lambda x: x[prop])


def create_df_from_xml(metadata_dir: str, output_dir: str, force: bool = False):
    filename = 'metadata'
    outfile = Path(output_dir) / f'{filename}.hdf5'
    if outfile.exists() and not force:
        LOG.info(f'Loading from HDF5: {outfile}')
        return rts.utils.dataframe_from_hdf5(output_dir, filename)

    LOG.info(f'Building dataframe from XMLs')
    mapping = parse_data_mapping(os.path.join(metadata_dir, 'fieldmapping.xml'))
    simple_mapping = filter_mapping(mapping)
    items = parse_all_xml_files(metadata_dir, simple_mapping)
    df = items_to_dataframe(items)
    df = process_raw_dataframe(df)
    rts.utils.dataframe_to_hdf5(output_dir, filename, df)
    return df


def match_video_files(df: pd.DataFrame, root_folder: str) -> pd.DataFrame:
    def find_mp4_files(root_folder: str):
        mp4_files = []
        for foldername, _, filenames in os.walk(root_folder):
            for filename in filenames:
                if filename.endswith('.mp4'):
                    mp4_files.append(os.path.join(foldername, filename))

        return mp4_files

    mp4_files = find_mp4_files(root_folder)
    file_map = {}
    for mp4 in mp4_files:
        file_map[mp4.split('/')[-1].split('.')[0]] = mp4
 
    LOG.info(f'Found {len(file_map)} mp4 files')
    df['path'] = df['guid'].apply(lambda x: file_map[x] if x in file_map else None)
    # # we need a path for each video, otherwise we can't process
    df = df[df['path'].notna()]
    return df.reset_index(drop=True)