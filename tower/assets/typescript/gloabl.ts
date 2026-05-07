
import { _decorator, Component, Node } from 'cc';
const { ccclass, property } = _decorator;

/**
 * Predefined variables
 * Name = Gloabl
 * DateTime = Wed Feb 16 2022 17:47:06 GMT+0800 (中国标准时间)
 * Author = tc123456
 * FileBasename = Gloabl.ts
 * FileBasenameNoExtension = Gloabl
 * URL = db://assets/typescript/Gloabl.ts
 * ManualUrl = https://docs.cocos.com/creator/3.4/manual/zh/
 *
 */
 
@ccclass('Gloabl')
export default class Gloabl{

    public static speed: number = 1000;

    public static coin: number = 30;

}

/**
 * [1] Class member could be defined like this.
 * [2] Use `property` decorator if your want the member to be serializable.
 * [3] Your initialization goes here.
 * [4] Your update function goes here.
 *
 * Learn more about scripting: https://docs.cocos.com/creator/3.4/manual/zh/scripting/
 * Learn more about CCClass: https://docs.cocos.com/creator/3.4/manual/zh/scripting/ccclass.html
 * Learn more about life-cycle callbacks: https://docs.cocos.com/creator/3.4/manual/zh/scripting/life-cycle-callbacks.html
 */
