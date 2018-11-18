<h2><i> Description </i></h2>

Algorithm for detecting apparent motion of image objects

<h2><i> Quick start </i></h2>

Project requires <b>opencv</b> and <b>numpy</b> libraries to run

Works stable with <b>Python 3.6.1</b>

<h2><i> Documentation </i></h2>

Algorithm steps
<ul>
  <li> capture initial frame </li>
  <li> search for points to track if needed </li>
  <li> capture another frame
  <li> calculate the difference between tracked points </li>
  <li> repeat the process exept the first step </li>
</ul>

The program displays the result as a video with marked tracing points. They are colored with blue. The clearer the motion the brighter the point
<p align="center"><b> Examples </b></p>
<p align="center">
  <image src="https://user-images.githubusercontent.com/29633052/48676463-5b969c00-eb78-11e8-9e80-335c99576821.png"></image>
</p>

Points on left are less bright because that part of the train moves visually slower

<p align="center">
  <image src="https://user-images.githubusercontent.com/29633052/48676532-2a6a9b80-eb79-11e8-959b-52662e6aae62.png"></image>
</p>
