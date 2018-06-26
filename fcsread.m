function [data,paramVals,textHeader] = fcsread(filename)
% FCSREAD read a FACS FCS format file.
%
%   DATA = FCSREAD(FILENAME) reads an FCS format file FILENAME, returning
%   the data in the file as an array with the data for each parameter
%   stored in the columns of the array. 
%
%   [DATA, PARAMS] = FCSREAD(FILENAME) returns information about the
%   parameters, such as the name, range and amplification factors in a
%   structure.
%
%   [DATA, PARAMS, HEADER] = FCSREAD(FILENAME) returns the TEXT field
%   header information in an Nx2 cell array with the field names in the
%   first column and the values in the second column.
%
%   Example:
%
%       % Read a sample FCS file
%       [data,params] = fcsread('SampleFACS.fcs');
%
%       % Display a histogram of the first parameter values
%       hist(data(:,1),max(data(:,1))-min(data(:,1)));
%       title(params(1).LongName);
%
%       % Create a scatter plot the data using a log scale
%       numDecades = params(2).Amplification(1);
%       theRange = params(2).Range;
%       semilogy(data(:,1),10.^(data(:,2)/(theRange/numDecades)),'+');
%       xlabel(params(1).Name); ylabel(params(2).Name);
%

%   Copyright 2005 The MathWorks, Inc.
%   $Revision:  $  $Date: 2005/3/24 20:41:57 $

% FCS format specified here:
% http://www.isac-net.org/resources/fcs3.htm

% open the file...

correctEndian = false;
endianCount = 1;
endianOptions = {'l','b'};
while correctEndian == false
    try
        fid = fopen(filename,'rb',endianOptions{endianCount});
        if fid == -1
            if ~exist(filename)
                error('fcsread:CannotFindFile','%s does not appear to exist.',filename)
            end
        end
    catch
        if ~ischar(filename)
            error('fcsread:InvalidInput','Input must be a character array')
        end

        if endianCount == 3
            error('fcsread:CannotDetermineEndian','%s does not appear to be a valid FCS file.',filename)
        end
        rethrow(lasterr);
    end


    % read the header
    fileHeader = readHeader(fid,filename);

    % Check endian-ness
    correctEndian = checkByteOrder(fileHeader.textData,endianCount);
    if ~correctEndian
        fclose(fid);
        endianCount = endianCount+1;
    end
end

% get the number of parameters

textData = strrep(fileHeader.textData,char(0),'');
textCell = strread(textData,'%s','delimiter','\\');
if isempty(textCell{1})
    textCell(1) = [];
end
textHeader = reshape(textCell,2,numel(textCell)/2)';
numValues = lookupNumericData(textCell,'$TOT');

% get details for each parameter
[numParams, paramVals] = getParamVals(textCell);

% read the data
if any(diff([paramVals.Bits]))
    error('fcsread:MixedIntegersNotSupported',...
        'Data in %s is stored in an unsupported format.',filename)
end
checkInts = strcmpi('I',lookupTextData(textCell,'$DATATYPE'));
if ~checkInts
    error('fcsread:NonIntegersNotSupported',...
        'Data in %s is stored in an unsupported format.',filename)
end
checkMode = strcmpi('L',lookupTextData(textCell,'$MODE'));
if ~checkMode
    error('fcsread:NonListNotSupported',...
        'Data in %s is stored in an unsupported format.',filename)
end

intSize = paramVals(1).Bits;
dataFormat = sprintf('uint%d',intSize);

% for now we only support one size of uints in a binary list.
fseek(fid,fileHeader.dataStart,-1);
data = fread(fid,numValues*numParams,dataFormat);
fclose(fid);
% Check we read the right amount of data
if numel(data) ~= numValues*numParams
    error('fcsread:DataSizeMismatch',...
        'Error reading the correct number of data values.')
end

% reshape the data into columns
data = reshape(data,numParams,numValues)';

% check ranges
maxParams = max(data);
paramRanges = [paramVals.Range];
if any(maxParams > paramRanges)
    error('fcsread:DataRangeError',...
        'Error reading the correct number of data values.')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [numParams, paramVals] = getParamVals(textCell);
numParams = lookupNumericData(textCell,'$PAR');


for count = 1:numParams
    paramVals(count).Name = lookupTextData(textCell,sprintf('$P%dN',count));
    paramVals(count).Range = lookupNumericData(textCell,sprintf('$P%dR',count));
    paramVals(count).Bits= lookupNumericData(textCell,sprintf('$P%dB',count));
    paramVals(count).Amplification = lookupNumericData(textCell,sprintf('$P%dE',count));
    paramVals(count).LongName= lookupTextData(textCell,sprintf('$P%dS',count));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function header = readHeader(fid,filename)
% Extract the header information

% we could check here for
header.ver = fread(fid,6,'*char')';
if isempty(strfind(header.ver,'FCS'))
    error('fcsread:NoFCSTag','%s does not appear to be a valid FCS file.',filename)
end
header.skip = fread(fid,4,'*char');
header.textStart = str2num(fread(fid,8,'*char')');
header.textEnd = str2num(fread(fid,8,'*char')');
header.dataStart = str2num(fread(fid,8,'*char')');
header.dataEnd = str2num(fread(fid,8,'*char')');
header.analysisStart = str2num(fread(fid,8,'*char')');
header.analysisEnd = str2num(fread(fid,8,'*char')');
fseek(fid,header.textStart,-1);
header.textData = fread(fid,header.textEnd -header.textStart+1,'*char')';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function checkEndian = checkByteOrder(textData,endianCount);
% check the endianness of the file.
if endianCount == 1
    checkEndian = any(strfind(textData,'1,2,3,4'));
else
    checkEndian = any(strfind(textData,'4,3,2,1'));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function val = lookupNumericData(textCell,fieldName)

val = find(strcmpi(textCell,fieldName));
if ~isempty(val)
    val = str2num(textCell{val+1});
else
    val = NaN;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function val = lookupTextData(textCell,fieldName)

val = find(strcmpi(textCell,fieldName));
if ~isempty(val)
    val = textCell{val+1};
else
    val = '';
end