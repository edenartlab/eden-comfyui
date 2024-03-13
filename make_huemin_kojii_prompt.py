def generate_kojii_huemin_prompt():

    # TODO: modify so that climate, landform and body_of_water are user inputs

    keywords = {
        "climate":       ["", "arid", "temperate", "tropical", "alpine", "cold", "warm", "humid", "dry", "mediterranean", "oceanic", "continental", "polar", "subtropical", "desert", "savanna", "rainforest", "tundra", "monsoon", "steppe"],
        "landform":      ["", "mountains", "valleys", "plateaus", "hills", "plains", "dunes", "canyons", "cliffs", "caves", "volcanoes", "rivers", "lakes", "icebergs", "fjords", "deltas", "estuaries", "wetlands", "deserts", "craters", "atolls", "peninsula", "islands surrounded by water", "basins", "gorges", "waterfalls", "rift valleys", "obsidian lava flows steam"],
        "body_of_water": ["", "oceans", "seas", "rivers", "lakes", "ponds", "streams", "creeks", "estuaries", "fjords", "bays", "gulfs", "lagoons", "marshes", "swamps", "reservoirs", "waterfalls", "glacial lakes", "wetlands", "springs", "brooks"],
        "structures":    ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "small bridges", "small tunnels", "small dams", "small skyscrapers", "small castles", "small temples", "small churches", "small mosques", "small fortresses", "small monuments", "small statues", "small towers", "small silos", "small industrial factories", "small piers", "small harbors"],
        "seasons":       ["", "spring", "summer", "autumn", "winter", "rainy", "sunny", "clouds from above", "stormy clouds from above", "foggy mist", "snowy", "windy", "humid", "dry", "hot", "cold", "mild", "freezing", "hail", "sleet", "blizzard", "heatwave", "drought"],
        "time_of_day":   [""],
        "colors":        ["", "monochromatic", "analogous", "complementary", "split-complementary", "triadic", "tetradic", "square", "neutral", "pastel", "warm", "cool", "vibrant", "muted", "earth tones", "jewel tones", "metallic"]
    }
    
    base_prompt = "isometric generative landscape orthographic abstract aj casson perlin noise 3d shaders areal embroidery minimalism claude monet oil painting pastel"
    
    # Randomly select one keyword from each category in the JSON data
    selected_climate       = random.choice(keywords['climate'])
    selected_landform      = random.choice(keywords['landform'])
    selected_body_of_water = random.choice(keywords['body_of_water'])
    selected_structure     = random.choice(keywords['structures'])
    selected_season        = random.choice(keywords['seasons'])
    selected_time_of_day   = random.choice(keywords['time_of_day'])
    selected_colors        = random.choice(keywords['colors'])

    # Construct a list of the selected keywords
    selected_keywords = [selected_climate, selected_landform, selected_body_of_water, selected_structure, selected_season, selected_time_of_day, selected_colors]
    landscape_keywords = " ".join(selected_keywords)

    # Construct the final prompt
    prompt = base_prompt + " (((" + landscape_keywords + ")))"
    return prompt