.. _sales-overseas-basicinfo-get_guarantor:

============================================
Sphinx Test
============================================


API概要
============================
* This is a bulleted list.
* It has two items, the second
  item uses two lines.

1. This is a numbered list.
2. It has two items too.

#. This is a numbered list.
#. It has two items too.



* this is
* a list

  * with a nested list
  * and some subitems

* and here the parent list continues



term (up to a line of text)
   Definition of the term, which must be indented

   and can even consist of multiple paragraphs

next term
   Description.



| These lines are
| broken exactly like in
| the source file.



This is a normal text paragraph. The next paragraph is a code sample::

   It is not processed in any way, except
   that the indentation is removed.

   It can span multiple lines.

This is a normal text paragraph again.



>>> 1 + 1
2



+------------------------+------------+----------+----------+
| Header row, column 1   | Header 2   | Header 3 | Header 4 |
| (header rows optional) |            |          |          |
+========================+============+==========+==========+
| body row 1, column 1   | column 2   | column 3 | column 4 |
+------------------------+------------+----------+----------+
| body row 2             | ...        | ...      |          |
+------------------------+------------+----------+----------+



Important

There must be a space between the link text and the opening < for the URL.



This is a paragraph that contains `a link`_.

.. _a link: https://domain.invalid/



=================
This is a heading
=================



:fieldname: Field content



.. function:: foo(x)
              foo(y, z)
   :module: some.module.name

   Return a line of text input from the user.



.. image:: _static/img/image.png



Lorem ipsum [#f1]_ dolor sit amet ... [#f2]_

.. rubric:: Footnotes

.. [#f1] Text of the first footnote.
.. [#f2] Text of the second footnote.



Lorem ipsum [Ref]_ dolor sit amet.

.. [Ref] Book or article reference, URL or whatever.



.. |name| replace:: replacement *text*


|name|


.. |caution| image:: _static/img/image2.png
             :alt: Warning!


|caution|


.. This is a comment.



..
   This whole indented block
   is a comment.

   Still in the comment.



.. function:: install()

   This function installs a `handler` for every signal known by the
   `signal` module.  See the section `about-signals` for more information.




.. _my-reference-label:

Section to cross-reference
============================

This is the text of the section.

It refers to the section itself, see :ref:`my-reference-label`.

:ref:`my-reference-label`



.. _my-figure:

.. figure:: _static/img/image2.png

   Figure caption



See :download:`this example script <_static/file/Dive into Python3-r802.pdf>`.



.. only:: builder_html

   See :download:`this example script <_static/file/Dive into Python3-r802.pdf>`.



Since Pythagoras, we know that :math:`a^2 + b^2 = c^2`.



:abbr:`LIFO (last-in, first-out)`.



... is installed in :file:`/usr/lib/python2.{x}/site-packages` ...



:kbd:`Control-x Control-f`.



:mailheader:`Content-Type`.



:manpage:`ls(1)`.



:menuselection:`Start --> Programs`



:samp:`print 1+{variable}`



:py:mod:`signal`



:mod:`signal`



.. note::

   This function is not suitable for sending spam e-mails.



.. versionadded:: 2.5
   The *spam* parameter.



.. deprecated:: 3.1
   Use :func:`spam` instead.



.. seealso::

   Module :py:mod:`zipfile`
      Documentation of the :py:mod:`zipfile` standard module.

   `GNU tar manual, Basic Tar Format <http://link>`_
      Documentation for tar archive files, including GNU tar extensions.



.. seealso:: modules :py:mod:`zipfile`, :py:mod:`tarfile`



.. centered:: LICENSE AGREEMENT



.. hlist::
   :columns: 2

   * A list of
   * short items
   * that should be
   * displayed
   * horizontally



.. highlight:: c



.. highlight:: python
   :linenothreshold: 5



.. code-block:: ruby

   Some Ruby code.



.. code-block:: ruby
   :linenos:

   Some more Ruby code.



.. code-block:: ruby
   :lineno-start: 10

   Some more Ruby code, with line numbering starting at 10.



.. code-block:: python
   :emphasize-lines: 3,5

   def some_function():
       interesting = False
       print 'This line is highlighted.'
       print 'This one is not...'
       print '...but this one is.'



.. code-block:: python
   :caption: this.py
   :name: this-py

   print 'Explicit is better than implicit.'



.. code-block:: ruby
   :dedent: 4

       some ruby code






.. glossary::

   environment
      A structure where information about all documents under the root is
      saved, and used for cross-referencing.  The environment is pickled
      after the parsing stage, so that successive runs only need to read
      and parse new and changed documents.

   source directory
      The directory which, including its subdirectories, contains all
      source files for one Sphinx project.



.. glossary::

   term 1
   term 2
      Definition of both terms.



.. glossary::

   term 1 : A
   term 2 : B
      Definition of both terms.



.. sectionauthor:: Guido van Rossum <guido@python.org>



.. index::
   single: execution; context
   module: __main__
   module: sys
   triple: module; search; path

The execution context
---------------------

...



.. index:: Python



.. index:: ! Python



.. index:: BNF, grammar, syntax, notation



This is a normal reST :index:`paragraph` that contains several
:index:`index entries <pair: index; entry>`.



.. only:: html and draft



.. math::

   (a + b)^2 = a^2 + 2ab + b^2

   (a - b)^2 = a^2 - 2ab + b^2



.. math::

   (a + b)^2  &=  (a + b)(a + b) \\
              &=  a^2 + 2ab + b^2



.. math:: (a + b)^2 = a^2 + 2ab + b^2



.. math::
   :nowrap:

   \begin{eqnarray}
      y    & = & ax^2 + bx + c \\
      f(x) & = & x^2 + 2xy + y^2
   \end{eqnarray}






.. py:function:: spam(eggs)
                 ham(eggs)

   Spam or ham the foo.



.. py:function:: filterwarnings(action, message='', category=Warning, \
                            module='', lineno=0, append=False)
    :noindex:



The function :py:func:`spam` does a similar thing.



.. function:: pyfunc()

   Describes a Python function.

Reference to :func:`pyfunc`.



.. py:function:: Timer.repeat(repeat=3, number=1000000)



.. py:class:: Foo

   .. py:method:: quux()

-- or --

.. py:class:: Bar

.. py:method:: Bar.quux()



.. py:decorator:: removename

   Remove name of the decorated function.

.. py:decorator:: setnewname(name)

   Set name of the decorated function to *name*.



.. py:function:: compile(source : string, filename, symbol='file') -> ast object



.. py:function:: send_message(sender, recipient, message_body, [priority=1])

   Send a message to a recipient

   :param str sender: The person sending the message
   :param str recipient: The recipient of the message
   :param str message_body: The body of the message
   :param priority: The priority of the message, can be a number 1-5
   :type priority: integer or None
   :return: the message id
   :rtype: int
   :raises ValueError: if the message_body exceeds 160 characters
   :raises TypeError: if the message_body is not a basestring



.. option:: dest_dir

   Destination directory.

.. option:: -m <module>, --module <module>

   Run a module as a script.



.. program:: rm

.. option:: -r

   Work recursively.

.. program:: svn

.. option:: -r revision

   Specify the revision to work upon.



.. describe:: PAPER

   You can set this variable to select a paper size.



.. js:function:: $.getJSON(href, callback[, errback])

   :param string href: An URI to the location of the resource.
   :param callback: Gets called with the object.
   :param errback:
       Gets called in case the request fails. And a lot of other
       text so we need multiple lines.
   :throws SomeError: For whatever reason in that case.
   :returns: Something.



.. js:class:: MyAnimal(name[, age])

   :param string name: The name of the animal
   :param number age: an optional age for the animal



.. rst:directive:: foo

   Foo description.

.. rst:directive:: .. bar:: baz

   Bar description.



.. rst:directive:: toctree

   .. rst:directive:option:: caption: caption of ToC

   .. rst:directive:option:: glob



.. rst:directive:: toctree

   .. rst:directive:option:: maxdepth
      :type: integer or no value



.. rst:role:: foo

   Foo description.



.. rst:role:: math:numref



.. math:: e^{i\pi} + 1 = 0
   :label: euler

Euler's identity, equation :math:numref:`euler`, was elected one of the
most beautiful mathematical formulas.


.. raw:: html

   <i class="fa fa-inbox"></i>




.. |ex1| replace:: 例1


.. |inbox| raw:: html

   <i class="fa fa-wallet"></i>



oo |inbox| kk





List Tables
-----------

.. list-table:: List tables can have captions like this one.
    :widths: 10 5 10 50
    :header-rows: 1
    :stub-columns: 1

    * - List table
      - Header 1
      - Header 2
      - Header 3 long. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nam sit amet mauris arcu.
    * - Stub Row 1
      - Row 1
      - Column 2
      - Column 3 long. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nam sit amet mauris arcu.
    * - Stub Row 2
      - Row 2
      - Column 2
      - Column 3 long. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nam sit amet mauris arcu.
    * - Stub Row 3
      - Row 3
      - Column 2
      - Column 3 long. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nam sit amet mauris arcu.






Code with Sidebar
-----------------

.. sidebar:: A code example

    With a sidebar on the right.2

