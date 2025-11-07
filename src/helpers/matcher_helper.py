from src.matching.matcher_subclasses import SynCompanyMatcher, WDCMatcher, CameraMatcher, MonitorMatcher, MusicBrainzMatcher

matchers_dict = {
    'synthetic_companies': SynCompanyMatcher,
    'wdc': WDCMatcher,
    'camera': CameraMatcher,
    'monitor': MonitorMatcher,
    'musicbrainz': MusicBrainzMatcher,
    # Add other matchers here
}
