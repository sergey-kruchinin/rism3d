import numpy as np
import re


def convertFormat( string ) :
    """Very simple converter from Fortran format line into Python one. 
    Known issues: cannot parse strings with custom exponent field width 
    like 1E15.5E5"""
    typeToken = '[IiAaLlEeFf]'
    fortran2pythonTypeConversion = {
        'I' :   int,
        'i' :   int,
        'A' :   lambda x: x.strip(),
        'a' :   lambda x: x.strip(),
        'L' :   bool,
        'l' :   bool,
        'E' :   float,
        'e' :   float,
        'F' :   float,
        'f' :   float,
    }
    widthToken = '^\d+'
    counterToken = '\d+$'
    t = re.search( typeToken, string )
    if t :
        w = re.search( widthToken, string[t.span()[1]:] )
        c = re.search( counterToken, string[:t.span()[0]] )
        if not w or not c :
            raise LookupError( 'cannot parse format string {0}'.format( string ) )
    else :
        raise LookupError( 'cannot parse format string {0}'.format( string ) )
    result = {}
    result['TYPE'] = fortran2pythonTypeConversion[t.group()]
    result['COUNTER'] = int( c.group() )
    result['WIDTH'] = int( w.group() )
    return result 


convertFormatTests = [
    ( '1P5E24.16',   {'COUNTER': 5, 'WIDTH': 24, 'TYPE': float} ),
    ( '10I8',        {'COUNTER': 10, 'WIDTH': 8, 'TYPE': int} ),
#    ( '20A4',        {'COUNTER': 20, 'WIDTH': 4, 'TYPE': lambda x: x} ),
#    ( '20a4',        {'COUNTER': 20, 'WIDTH': 4, 'TYPE': <function mdl.convertFormat.<locals>.<lambda>>} ),
    ( '1P5e24.16',   {'COUNTER': 5, 'WIDTH': 24, 'TYPE': float} ),
    ( '5E16.8',      {'COUNTER': 5, 'WIDTH': 16, 'TYPE': float} ),
]


class amberParm( dict ) :
    def __init__( self, filename, sections=None ) :
        sectionName = ''
        skipSection = False
        fmt = None
        self.version = ''
        self.date = ''

        def createSection() :
            nonlocal sectionName
            nonlocal fmt
            nonlocal skipSection
            sectionName = line.replace( '%FLAG', '' ).strip()
            fmt = None
            if sections and ( not sectionName in sections ) :
                skipSection = True
                return
            skipSection = False
            if sectionName in self.keys() :
                print( 'WARNING: Repeated declaration of section ' + sectionName )
            self[sectionName] = np.empty( 0, dtype=int )    

        def setFormat() :
            nonlocal fmt
            nonlocal skipSection
            if skipSection :
                return
            if fmt :
                print( 'WARNING: Reseting format for section ' + sectionName )
            fortranFmt = re.search( '(?<=%FORMAT\().+(?=\))', line ).group() 
            fmt = convertFormat( fortranFmt )

        def getData() :
            nonlocal skipSection
            if skipSection :
                return
            w = fmt['WIDTH']
            c = min( len( line ) - 1, fmt['COUNTER'] * fmt['WIDTH'] )
            dataChunk = [fmt['TYPE']( line[i:i+w] ) for i in range( 0, c, w )]
            self[sectionName] = np.append( self[sectionName], dataChunk )

        def getVersionAndDate() :
            version = re.search( '(?<=VERSION_STAMP =)\s*V\d*\.\d*', line )
            if version :
                self.version = version.group().strip()
            date = re.search( '(?<=DATE =).*$', line )
            if date :
                self.date = date.group().strip()

        with open(filename) as fin:
            for line in fin:
                token = re.search( '^%[A-Z]+', line )
                if not token :
                    getData()
                    continue
                if token.group() == '%VERSION' :
                    getVersionAndDate()
                    continue
                elif token.group() == '%FLAG' :
                    createSection()
                    continue
                elif token.group() == '%FORMAT' :
                    setFormat()
                    continue
                elif token.group() == '%COMMENT' :
                    continue
        
