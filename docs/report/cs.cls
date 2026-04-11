% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A new style for Cerebras Systems pre-prints
% Author: omid.solari@cerebreas.net
% Version: 1.2
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Class declarations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{cs}
\LoadClassWithOptions{article}

% Layout %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage[top=2.25cm, bottom=2.5cm, left=2.5cm, right=2.5cm, columnsep=0.65cm]{geometry}
% microtype with expansion disabled: cmvtt (used for \ttdefault below) is a
% bitmap font that cannot be expanded.  Protrusion is still active.
\RequirePackage[protrusion=true,expansion=false]{microtype}
\RequirePackage{placeins}
\RequirePackage{hyphenat}
\RequirePackage{setspace}
\RequirePackage{parskip}
\RequirePackage[english]{babel}
\RequirePackage{etoolbox}

\DisableLigatures[f]{family=sf*}

% Graphics %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{graphicx}
\RequirePackage{subcaption}

% Tables %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{booktabs}
\RequirePackage{nicematrix}
\RequirePackage{multirow}
\RequirePackage{bm}
\newcommand{\nm}[1]{#1}

% Colorful stuff %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage[most]{tcolorbox}
\RequirePackage{xcolor}
\definecolor{metablue}{HTML}{0064E0}
\definecolor{metafg}{HTML}{1C2B33}
\definecolor{metabg}{HTML}{F1F4F7}

\definecolor{csrust}{HTML}{F1592A}
\definecolor{csfg}{HTML}{000000}
\definecolor{csbg}{HTML}{FCDDD4}

% References %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{hyperref}
\RequirePackage[noabbrev,nameinlink]{cleveref}
\hypersetup{
  colorlinks,
  linkcolor=csrust,
  citecolor=csrust,
  urlcolor=csrust
}

\RequirePackage[round,authoryear]{natbib}
\def\bibfont{\small}

% Section and caption format %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{titlesec}
\titleformat*{\paragraph}{\itshape}
\titleformat*{\section}{\Large\sffamily\bfseries}
\titleformat*{\subsection}{\large\sffamily\bfseries}
\titleformat*{\subsubsection}{\normalsize\sffamily\bfseries}
\DeclareTextFontCommand{\textbf}{\bfseries\sffamily}

\RequirePackage{caption}
\DeclareCaptionLabelSeparator{custom}{}
\DeclareCaptionFormat{custom}{{\sffamily\textbf{#1 #2}} #3}
\captionsetup{singlelinecheck=false,format=custom,labelsep=custom,font=small}
\captionsetup[sub]{singlelinecheck=true,format=custom,labelsep=custom,font=small}

% Meta custom fonts %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NOTE (portability): keep a standard sans-serif fallback for broad pdfLaTeX compatibility.

\renewcommand*\ttdefault{cmvtt}
\RequirePackage[T1]{fontenc}
\renewcommand{\sfdefault}{cmss}

% Authors, affiliations, infos %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\makeatletter
\renewcommand\addtolist[5][]{%
  \begingroup
    \if\relax#3\relax\def\sep{}\else\def\sep{#5}\fi
    \let\protect\@unexpandable@protect
    \xdef#3{\expandafter{#3}\sep #4[#1]{#2}}%
  \endgroup
}
\makeatother

\newcommand\authorlist{}
\newcommand\authorformat[2][]{{\sffamily \bfseries #2$^{#1}$}}
\renewcommand\author[2][]{\addtolist[#1]{#2}{\authorlist}{\authorformat}{, }}

\newcommand\affiliationlist{}
\newcommand\affiliationformat[2][]{{\normalsize $^{#1}$#2}}
\newcommand\affiliation[2][]{\addtolist[#1]{#2}{\affiliationlist}{\affiliationformat}{, }}

\newcommand\contributionlist{}
\newcommand\contributionformat[2][]{{\small $^{#1}$#2}}
\newcommand\contribution[2][]{\addtolist[#1]{#2}{\contributionlist}{\contributionformat}{, }}

\newcommand\metadatalist{}
\newcommand\metadataformat[2][]{{\small {\sffamily \bfseries #1:} #2}}
\newcommand\metadata[2][]{\addtolist[#1]{#2}{\metadatalist}{\metadataformat}{\par}}

\newcommand{\abstractlist}{}
\renewcommand{\abstract}[1]{\gdef\abstractlist{{\color{csfg} #1}}}
\newcommand{\email}[1]{\href{mailto:#1}{\texttt{#1}}}

\renewcommand\date[1]{\metadata[Date]{#1}}
\newcommand\correspondence[1]{\metadata[Correspondence]{#1}}

\renewcommand{\title}[1]{\newcommand{\titlelist}{{\huge\sffamily\bfseries #1}}}

\newcommand{\beginappendix}{\appendix{\huge\sffamily Appendix\par}}

% Titlebox %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand{\mymaketitle}{%
  \tcbset{
    enhanced,
    frame hidden,
    colback=csbg,
    arc=10pt,
    left=0.25cm,
    right=0.25cm,
    top=0.25cm,
    bottom=0.95cm,
    before skip=0pt,
    after skip=0pt,
    grow to left by=1.5pt,
    grow to right by=1.5pt,
    breakable,
    pad at break*=1mm,
    segmentation hidden,
    overlay unbroken={\node[
      anchor=south east,
      at=(frame.south east),
      xshift=-0.45cm,
      yshift=0.20cm] {\includegraphics[width=1.65cm]{assets/cerebras_logo.png}};},
    overlay first={},
    overlay middle={},
    overlay last={\node[
      anchor=south east,
      at=(frame.south east),
      xshift=-0.45cm,
      yshift=0.20cm] {\includegraphics[width=1.65cm]{assets/cerebras_logo.png}};}
  }
  \begin{tcolorbox}
    \setlength{\parindent}{0cm}
    \setlength{\parskip}{0.42cm}
    {
      \setlength{\parskip}{0cm}
      \raggedright
      \nohyphens
      {
        \setstretch{1.45}
        \titlelist\par
      }
      \vskip 0.16cm
      \authorlist\par
      \vskip 0.16cm
      \affiliationlist\par
      \contributionlist\par
    }
    % \vskip 0.28cm
    % {\sffamily\bfseries Abstract}\par
    \vskip 0.14cm
    \abstractlist\par
    \vskip 0.34cm
    {
      \setlength{\parskip}{0cm}
      \ifdefempty{\metadatalist}{}{\metadatalist\par}
    }
  \end{tcolorbox}
  \tcbset{reset}
  \FloatBarrier
}

\if@twocolumn
\renewcommand{\maketitle}{%
  \onecolumn
  \mymaketitle
  \vskip 0.38cm
  \twocolumn
}
\else
\renewcommand{\maketitle}{%
  \mymaketitle
}
\fi
