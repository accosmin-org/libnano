set nocompatible

syntax enable

set tabstop=4           " The width of a TAB
set shiftwidth=4        " Indent size
set softtabstop=4       " Sets the number of columns for a TAB
set expandtab           " Expand TABs to spaces
set smarttab            " Make "tab" insert indents instead of tabs at the beginning of a line
set exrc
set secure

set number              " Show line numbers
set showcmd             " Show command in bottom bar
set nocursorline        " Highlight current line
set wildmenu
"set lazyredraw
set showmatch           " Higlight matching parenthesis

set incsearch           " search as characters are entered
set hlsearch            " highlight matches

" Replace tabs with spaces
map <F2> :retab <CR> :w <CR>

" Toggle whitespace visibility
nmap <F3> :set list!<CR>
set listchars=tab:>-,trail:-

" Line extend: http://www.alexeyshmalko.com/2014/using-vim-as-c-cpp-ide/
set colorcolumn=120
highlight ColorColumn ctermbg=darkgray

" Text search
highlight Search cterm=NONE ctermfg=NONE ctermbg=darkgray

set path+=src
set path+=app
set path+=test
set path+=docs
set path+=cmake
set path+=include
set path+=example
set path+=scripts

" Toggle between header and implementation
" map <F4> :e %:p:s,.h$,.X123X,:s,.cpp$,.h,:s,.X123X$,.cpp,<CR>

function! Mosh_Flip_Ext()
    " Switch editing between .c* and .h* files (and more).
    " Since .h file can be in a different dir, call find.
    if match(expand("%"),'\.cpp') > 0
        let s:flipname = substitute(expand("%"),'\.cpp\(.*\)','.h\1',"")
        exe ":e " s:flipname
    elseif match(expand("%"),"\\.h") > 0
        let s:flipname = substitute(expand("%"),'\.h\(.*\)','.cpp\1',"")
        exe ":e " s:flipname
    endif
endfun

map <F4> :call Mosh_Flip_Ext()<CR>

" Trim trailing whitespaces when saving
function! StripTrailingWhitespaces()
    let l = line(".")
    let c = col(".")
    %s/\s\+$//e
    call cursor(l, c)
endfunction
autocmd BufWritePre     * :call StripTrailingWhitespaces()

function! Formatonsave()
  let l:formatdiff = 10
  if filereadable("/usr/share/clang/clang-format.py")
      py3f /usr/share/clang/clang-format.py
  endif
  if filereadable("/usr/share/clang/clang-format-18/clang-format.py")
      py3f /usr/share/clang/clang-format-18/clang-format.py
  endif
endfunction
autocmd BufWritePre *.h,*.cc,*.cpp call Formatonsave()
